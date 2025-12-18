from nilearn.image import resample_img
import nibabel as nib
import os
import glob
import numpy as np

import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import glob


def resample_to_mni152(data_path, template_path, output_path=None):
    data = nib.load(data_path)
    template = nib.load(template_path)
    resampled = resample_img(data, target_affine=template.affine, target_shape=(91,109,91), interpolation="linear")
    if output_path is not None:
        nib.save(resampled, output_path)

def convert_fMRIvols_to_A424(data_path, atlas_path, nParcels=424):
    dts = nib.load(data_path).get_fdata()
    label = nib.load(atlas_path).get_fdata()
    m = dts.reshape((-1, dts.shape[-1])).T
    sh = m.shape[0]

    pMeas = np.zeros((nParcels, 3))
    pmTS = np.zeros((sh, nParcels))

    for i in range(1, nParcels + 1):
        ind = (label == i)
        y = m[:, ind]
        pmTS[:, i - 1] = np.nanmean(y, axis=1)

    # Replace NaNs with 0
    pmTS[np.isnan(pmTS)] = 0

    return pmTS

def convert_to_arrow(args):
    """
    Arguments:
        args: arguments object containing parameters:
            --ts_data_dir
            --dataset_name
            --metadata_path
            --save_dir
        final save path: concatenation of dataset save directory and arrow dataset name
    """
    save_path = os.path.join(args["save_dir"], args["dataset_name"])
    # --- Train/val/test Split ---#
    print("fMRI Data Arrow Conversion Starting...")
    # Assuming that filename is patient ID, thus each file with unique name is a separate patient.
    all_dat_files = os.listdir(args["ts_data_dir"])
    all_dat_files = [filename for filename in all_dat_files if ".dat" in filename]
    if args["metadata_path"] is not None:
        metadata = pd.read_csv(args["metadata_path"])

    try: 
        all_dat_files.remove("A424_Coordinates.dat")
        print('A424_Coordinates was removed from the list')
    except ValueError:
        print("There's no A424 Coordinates dat file")
    all_dat_files.sort()

    train_split_idx = len(all_dat_files)
    train_files = all_dat_files[:train_split_idx]
    sh_35 = 0
    sh_less_200 = 0
    for idx,file in enumerate(tqdm(all_dat_files)):
        try:
            sample = np.loadtxt(os.path.join(args["ts_data_dir"],file)).T
            if sample.shape[0] < 200:
                print(sample.shape, idx, "ommitted due to insufficient data")
                sh_less_200 += 1
            else:
                sh_35 += 1

        except UnicodeDecodeError:
            print(file)
        

    print(f"Not processing {sh_less_200} files due to insufficient fMRI data")
    compute_Stats=True
    if compute_Stats: 
        num_files = sh_35 #len(all_dat_files_rs) + len(all_dat_files_tf)
        all_stds = np.zeros([num_files, 424]) 
        all_data = np.empty([num_files*200, 424])
        for idx,file in enumerate(tqdm(train_files)):
            if idx == num_files:
                break

            try:
                sample = np.loadtxt(os.path.join(args["ts_data_dir"],file)) #490, 424

            except UnicodeDecodeError:
                print(file)

            sample_mean = sample.mean(axis=0, keepdims=True)
            sample_mean = sample_mean[None,:].repeat(sample.shape[0],1).squeeze()
            sample = sample - sample_mean

            idx_sample=idx


            if sample.shape[0] < 200:
                sample = np.pad(sample, ((0, 200-sample.shape[0]),(0,0)), 'constant', constant_values=0)
            try:
                all_data[idx*200:(idx+1)*200,:] = sample[:200,:]
            except ValueError:
                print(sample.shape)
                print('idx: {}, idx_sample: {}'.format(idx,idx_sample))

        global_std = np.std(all_data, axis=0) 
        data_median_per_voxel = np.median(all_data,axis=0)
        data_mean_per_voxel = np.mean(all_data,axis=0)

        all_data_nonzeros = np.copy(all_data)
        all_data_nonzeros[all_data_nonzeros == 0] = 'nan'
        quartiles = np.nanpercentile(all_data_nonzeros, [25, 75], axis=0)
        IQR = quartiles[1,:]-quartiles[0,:]


    # --- Normalization Calculations ---#
    # Calculate min and max value across train, validation, and test sets
    global_train_max = -1e9
    global_train_min = 1e9
    voxel_maximums_train = []
    voxel_minimums_train = []

    for filename in tqdm(train_files, desc="Getting normalization stats"):
        dat_arr = np.loadtxt(os.path.join(args["ts_data_dir"], filename)).astype(
            np.float32
        )
        if dat_arr.shape[0] < 200:
            dat_arr = np.pad(dat_arr, ((0, 200 - dat_arr.shape[0]), (0, 0)), 'constant', constant_values=0)

        
        if np.max(dat_arr) > global_train_max:
            global_train_max = np.max(dat_arr)
        if np.min(dat_arr) < global_train_min:
            global_train_min = np.min(dat_arr)

        dat_arr_max = np.max(dat_arr, axis=1)
        dat_arr_min = np.min(dat_arr, axis=1)
        voxel_maximums_train.append(dat_arr_max)
        voxel_minimums_train.append(dat_arr_min)

    voxel_maximums_train = np.stack(voxel_maximums_train, axis=0)
    voxel_minimums_train = np.stack(voxel_minimums_train, axis=0)
    global_per_voxel_train_max = np.max(voxel_maximums_train, axis=0)
    global_per_voxel_train_min = np.min(voxel_minimums_train, axis=0)

    # --- Convert All .dat Files to Arrow Datasets ---#
    train_dataset_dict = {
        "Raw_Recording": [],
        "Voxelwise_RobustScaler_Normalized_Recording": [],
        "All_Patient_All_Voxel_Normalized_Recording": [],
        "Per_Patient_All_Voxel_Normalized_Recording": [],
        "Per_Patient_Per_Voxel_Normalized_Recording": [],
        "Per_Voxel_All_Patient_Normalized_Recording": [],
        "Subtract_Mean_Normalized_Recording": [],
        "Subtract_Mean_Divide_Global_STD_Normalized_Recording": [],
        "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording": [],
        "Filename": [],
        "Patient ID": [],
    }
    if args["metadata_path"] is not None:
        feature_list = metadata.columns.to_list()
        for k in feature_list:
            train_dataset_dict[k] = []
    

    for filename in tqdm(train_files, desc="Normalizing Data"):
        dat_arr = np.loadtxt(os.path.join(args["ts_data_dir"], filename)).astype(
            np.float32
        ).T

        if dat_arr.shape[0] < 200:
            dat_arr = np.pad(dat_arr, ((0, 200 - dat_arr.shape[0]), (0, 0)), 'constant', constant_values=0)

        if dat_arr.shape[0] > 424:
            dat_arr = dat_arr[:350, :]

        global_norm_dat_arr = np.copy(dat_arr)
        per_patient_all_voxels_norm_dat_arr = np.copy(dat_arr)
        per_patient_per_voxel_norm_dat_arr = np.copy(dat_arr)
        per_voxel_all_patient_norm_dat_arr = np.copy(dat_arr)
        recording_mean_subtracted = np.copy(dat_arr)
        recording_mean_subtracted2 = np.copy(dat_arr)
        recording_mean_subtracted3 = np.copy(dat_arr.T)
        global_std = 41.44047  # calculated in normalization notebook
        _99th_percentile = 111.13143061224855  # calculated externally

        # All patients, all voxels normalization
        if (global_train_max - global_train_min) > 0.0:
            global_norm_dat_arr = (global_norm_dat_arr - global_train_min) / (
                global_train_max - global_train_min
            )

        # Per patient all voxel normalization
        patient_all_voxel_min_val = np.min(per_patient_all_voxels_norm_dat_arr)
        patient_all_voxel_max_val = np.max(per_patient_all_voxels_norm_dat_arr)
        if (patient_all_voxel_max_val - patient_all_voxel_min_val) > 0.0:
            per_patient_all_voxels_norm_dat_arr = (
                per_patient_all_voxels_norm_dat_arr - patient_all_voxel_min_val
            ) / (patient_all_voxel_max_val - patient_all_voxel_min_val)

        # Per patient per voxel normalization
        for voxel_idx in range(dat_arr.shape[1]):
            patient_voxel_min_val = per_patient_per_voxel_norm_dat_arr[
                :, voxel_idx
            ].min()
            patient_voxel_max_val = per_patient_per_voxel_norm_dat_arr[
                :, voxel_idx
            ].max()
            if (patient_voxel_max_val - patient_voxel_min_val) > 0.0:
                per_patient_per_voxel_norm_dat_arr[:, voxel_idx] = (
                    per_patient_per_voxel_norm_dat_arr[:, voxel_idx]
                    - patient_voxel_min_val
                ) / (patient_voxel_max_val - patient_voxel_min_val)

        # Per voxel all patient normalization
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_maximum = global_per_voxel_train_max[voxel_idx]
            voxel_minimum = global_per_voxel_train_min[voxel_idx]
            if (voxel_maximum - voxel_minimum) > 0.0:
                per_voxel_all_patient_norm_dat_arr[:, voxel_idx] = (
                    per_voxel_all_patient_norm_dat_arr[:, voxel_idx] - voxel_minimum
                ) / (voxel_maximum - voxel_minimum)

        # Subtract Mean, Scale by Global Standard Deviation normalization
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_mean = recording_mean_subtracted[:, voxel_idx].mean()
            recording_mean_subtracted[:, voxel_idx] = (
                recording_mean_subtracted[:, voxel_idx] - voxel_mean
            )

        z_score_global_recording = np.divide(recording_mean_subtracted, global_std)

        # Subtract Mean, Scale by global 99th percentile
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_mean = recording_mean_subtracted2[:, voxel_idx].mean()
            recording_mean_subtracted2[:, voxel_idx] = (
                recording_mean_subtracted2[:, voxel_idx] - voxel_mean
            )
        
        #Voxelwise Robust Scaler Normalization
        recording_mean_subtracted3 = recording_mean_subtracted3 - recording_mean_subtracted3.mean(axis=0)
        recording_mean_subtracted3 = (recording_mean_subtracted3 - data_median_per_voxel / IQR)

        _99th_global_recording = np.divide(recording_mean_subtracted2, _99th_percentile)


        train_dataset_dict["Raw_Recording"].append(dat_arr)
        train_dataset_dict["Voxelwise_RobustScaler_Normalized_Recording"].append(recording_mean_subtracted3)
        train_dataset_dict["All_Patient_All_Voxel_Normalized_Recording"].append(
            global_norm_dat_arr
        )
        train_dataset_dict["Per_Patient_All_Voxel_Normalized_Recording"].append(
            per_patient_all_voxels_norm_dat_arr
        )
        train_dataset_dict["Per_Patient_Per_Voxel_Normalized_Recording"].append(
            per_patient_per_voxel_norm_dat_arr
        )
        train_dataset_dict["Per_Voxel_All_Patient_Normalized_Recording"].append(
            per_voxel_all_patient_norm_dat_arr
        )
        train_dataset_dict["Subtract_Mean_Normalized_Recording"].append(
            recording_mean_subtracted
        )
        train_dataset_dict[
            "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
        ].append(z_score_global_recording)
        train_dataset_dict[
            "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording"
        ].append(_99th_global_recording)
        train_dataset_dict["Filename"].append(filename)

        train_dataset_dict["Patient ID"].append(filename.split(".dat")[-1])
        if args["metadata_path"] is not None:
            feature_list = metadata.columns.to_list()
            for k in feature_list:
                idx = metadata[metadata["image_path"].apply(lambda x: os.path.basename(x)) == filename].index
                train_dataset_dict[k].append(metadata[k].iloc[idx].values)


    arrow_train_dataset = Dataset.from_dict(train_dataset_dict)
    arrow_train_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "data")
    )

    # --- Save Brain Region Coordinates Into Another Arrow Dataset ---#
    coords_dat = np.loadtxt(os.path.join("utils/atlases", "A424_Coordinates.dat")).astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "Brain_Region_Coordinates")
    )
    print("Done.")