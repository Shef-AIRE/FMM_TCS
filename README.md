# FMM_TCS
This is the official released repository for the manuscript "*Interpretable foundation-model-boosted multimodal learning facilitates precision medicine for neuropathic pain*".

## Environment preparation
To restore the environment we used in this study, run:
```
conda env create -f environment.yml
```

## MRI data preprocessing
### fMRIPrep

fMRIPrep is a standardized and reproducible preprocessing pipeline for functional MRI (fMRI) data. This guide describes how to run fMRIPrep using containerized workflows (Apptainer / Singularity / Docker), which is strongly recommended for both local and HPC environments.

---

#### 1. Install FreeSurfer and Obtain a License

fMRIPrep depends on FreeSurfer for anatomical surface reconstruction.

1. [Register and request a FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html)

2. After approval, download the `license.txt` file.

3. Place the license file in a persistent location, e.g.:
   ```bash
   ~/freesurfer/license.txt
   ```

4. You will later bind this file into the container.

Useful links:
- [FreeSurfer installation guide](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)

#### 2. Download the fMRIPrep Container

Choose the container system based on your environment.

##### Option A: Apptainer / Singularity (recommended for HPC)

```bash
apptainer pull docker://nipreps/fmriprep:24.1.0
```

or:

```bash
singularity pull docker://nipreps/fmriprep:24.1.0
```

This will generate:
```text
fmriprep_24.1.0.sif
```

##### Option B: Docker (local workstation)

```bash
docker pull nipreps/fmriprep:24.1.0
```

Useful links:
- [fMRIPrep installation documentation](https://fmriprep.org/en/stable/installation.html)


#### 3. Re-organize the Dataset into BIDS Format

fMRIPrep requires data to be organized following the Brain Imaging Data Structure (BIDS).

Example minimal structure:

```text
dataset/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── func/
│       ├── sub-01_task-rest_bold.nii.gz
│       └── sub-01_task-rest_bold.json
```

Key requirements:
- Proper BIDS naming conventions
- JSON sidecars with mandatory metadata (e.g., `RepetitionTime`)
- A valid `dataset_description.json`

Useful links:
- [BIDS specification](https://bids-specification.readthedocs.io/)
- [BIDS Starter Kit](https://bids-standard.github.io/bids-starter-kit/)
- [Online BIDS validator](https://bids-standard.github.io/bids-validator/)


#### 4. Run fMRIPrep Using a Shell Script

Example Apptainer-based script at `data_preprocessing/fmriprep.sh`.


### Resample to ICBM152 space
1. Download the ICBM152 template from [here](https://awesome.cs.jhu.edu/data/static/perm/MR-data/Atlas/mni_icbm152_nl_VI_nifti/) or use our demo template `atlases/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii`.
2. Run `resample_to_mni152` function in `data_preprocessing/helper.py`.

### Parcellation using AAL-424 atlas
1. Download the AAL-424 atlas [here](https://github.com/emergelab/hierarchical-brain-networks) or using our demo atlas `atlases/A424+2mm.nii.gz`.
2. Run `convert_fMRIvols_to_A424` function in `data_preprocessing/helper.py`.

### Convert to arrow dataset
The time-series data can be easily convert to the arrow format dataset using `convert_to_arrow` in `data_preprocessing/helper.py`.

Here is a demo script:
```python
args = {
    "ts_data_dir": "path_to_ts_dataset_folder",
    "dataset_name": "Name",
    "metadata_path": "path_to_metadata.csv",
    "save_dir": "directory_to_output_dataset"
}
```
The arrow dataset will be put under `save_dir/dataset_name` folder.

## Running model
### Download pretrained weights for `BrainLM`
The pretrained weights can be found [here](https://huggingface.co/vandijklab/brainlm).

Then, put it under `brainlm_mae/pretrained_models` folder.

## Run the model
Run `main.py` to train our model.

A quick start:
```python
python main.py --lable_name <the column name of label of your task in your metadata.csv, e.g., response>
```

## Acknowledgement
This repository is built upon [BrainLM](https://github.com/vandijklab/BrainLM?tab=readme-ov-file).
