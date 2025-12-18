import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pandas as pd

import torch


import random

from sklearn.preprocessing import StandardScaler



import warnings
warnings.filterwarnings("ignore")

def random_segment_padding(data, target_shape):
    """
    Pads the input tensor by sampling random segments from its content.
    
    Args:
        data (torch.Tensor): Input tensor of shape [424, 120].
        target_shape (tuple): Desired shape after padding [424, 120].
    
    Returns:
        torch.Tensor: Padded tensor of shape `target_shape`.
    """
    current_shape = data.shape
    
    # If no padding is needed, return the original tensor
    if current_shape == target_shape:
        return data

    padded_data = data.clone()  # Start with the original data
    padding_needed = target_shape[1] - current_shape[1]

    # Randomly sample columns for padding
    for _ in range(padding_needed):
        random_col = random.randint(0, current_shape[1] - 1)  # Random column index
        sampled_segment = data[:, random_col]  # Sample a column
        padded_data = torch.cat((padded_data, sampled_segment.unsqueeze(1)), dim=1)  # Add to data

    return padded_data

def preprocess_fmri(examples, coords_ds, recording_col_name, variable_of_interest_col_name="Response", moving_window_len=200, padding_mode="random_segment"):
    """
    Preprocessing function for dataset samples. This function is passed into Trainer as
    a preprocessor which takes in one row of the loaded dataset and constructs a model
    input sample according to the arguments which model.forward() expects.

    The reason this function is defined inside on main() function is because we need
    access to arguments such as cell_expression_vector_col_name.
    """
    label = examples[variable_of_interest_col_name][0]
    # brain_net = examples["Brain_Network"]
    if math.isnan(label):
        label = -1  # replace nans with -1
    # else:
    #     label = int(label
    label = torch.tensor(label, dtype=torch.int64)
    signal_vector = examples[recording_col_name]#[0]
    signal_vector = torch.tensor(signal_vector, dtype=torch.float32)
    # print(signal_vector.shape)
    # Choose random starting index, take window of moving_window_len points for each region
    if signal_vector.shape[1] > moving_window_len:
        start_idx = random.randint(0, signal_vector.shape[1] - moving_window_len)
        end_idx = start_idx + moving_window_len  # 24 patches per voxel, * 424 = 10176 total per sample
        signal_window = signal_vector[:, start_idx: end_idx]
        # print(start_idx, end_idx)
    elif signal_vector.shape[1] < moving_window_len:
        if padding_mode == "random_segment":
            signal_window = random_segment_padding(signal_vector, (signal_vector.shape[0], moving_window_len))
        else:
            signal_window = torch.nn.functional.pad(signal_vector, (0, moving_window_len - signal_vector.shape[1]), "constant", 0)
    else:
        signal_window = signal_vector
    
    
    # Append signal values and coords
    window_xyz_list = []
    for brain_region_idx in range(signal_window.shape[0]):
        # window_timepoint_list = torch.arange(0.0, 1.0, 1.0 / num_timepoints_per_voxel)

        # Append voxel coordinates
        xyz = torch.tensor([
            coords_ds[brain_region_idx]["X"],
            coords_ds[brain_region_idx]["Y"],
            coords_ds[brain_region_idx]["Z"]
        ], dtype=torch.float32)
        window_xyz_list.append(xyz)
    window_xyz_list = torch.stack(window_xyz_list)

    # Add in key-value pairs for model inputs which CellLM is expecting in forward() function:
    #  signal_vectors and xyz_vectors
    #  These lists will be stacked into torch Tensors by collate() function (defined above).
    examples["signal_vectors"] = signal_window.unsqueeze(0)
    examples["xyz_vectors"] = window_xyz_list.unsqueeze(0)
    # examples["brain_network"] = np.array(brain_net)
    examples["label"] = label
    return examples



class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, coords, recording_col_name, variable_of_interest_col_name, moving_window_len=200, structure_features="/home/wenrui/Projects/NeuroPain/fMRI/OpenNeuro/struct_stats.csv"):
        self.dataset = dataset
        self.coords_ds = coords
        self.struct_features_path = structure_features
        self.recording_col_name = recording_col_name
        self.variable_of_interest_col_name = variable_of_interest_col_name
        self.moving_window_len = moving_window_len
        
        # self.labels
        

        structure_feature_names = [
            "SubjectID",
            "Left-Lateral-Ventricle", 
            "Left-Inf-Lat-Vent", 
            "Left-Cerebellum-White-Matter", 
            "Left-Cerebellum-Cortex",
            "Left-Thalamus", 
            "Left-Caudate", 
            "Left-Putamen", 
            "Left-Pallidum", 
            "3rd-Ventricle", 
            "4th-Ventricle",
            "Brain-Stem", 
            "Left-Hippocampus", 
            "Left-Amygdala", 
            "CSF", 
            "Left-Accumbens-area", 
            "Left-VentralDC",
            "Left-vessel", 
            "Left-choroid-plexus", 
            "Right-Lateral-Ventricle", 
            "Right-Inf-Lat-Vent",
            "Right-Cerebellum-White-Matter", 
            "Right-Cerebellum-Cortex", 
            "Right-Thalamus", 
            "Right-Caudate",
            "Right-Putamen", 
            "Right-Pallidum", 
            "Right-Hippocampus", 
            "Right-Amygdala", 
            "Right-Accumbens-area",
            "Right-VentralDC", 
            "Right-vessel", 
            "Right-choroid-plexus", 
            "5th-Ventricle", 
            "WM-hypointensities",
            "Left-WM-hypointensities", 
            "Right-WM-hypointensities", 
            "non-WM-hypointensities",
            "Left-non-WM-hypointensities", 
            "Right-non-WM-hypointensities", 
            "Optic-Chiasm", "CC_Posterior",
            "CC_Mid_Posterior", 
            "CC_Central", 
            "CC_Mid_Anterior", 
            "CC_Anterior"
        ]
        if structure_features is not None:
            self.structure_features = pd.read_csv(structure_features)[structure_feature_names]
            # self.structure_features = torch.tensor(self.structure_features, dtype=torch.float32)
        else:
            self.structure_features = None
        self.features, self.labels, self.structure_features = self._load_data()
        scaler = StandardScaler()
        self.structure_features = scaler.fit_transform(self.structure_features)
        self.structure_features = torch.tensor(self.structure_features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    
    def _load_data(self):
        features = []
        labels = []
        struc_features = []
        for recording_idx in range(self.dataset.num_rows):
            example1 = self.dataset[recording_idx]
            if "vas" in self.variable_of_interest_col_name.lower():
                filter_flag = (example1["Response"][0] > 0)
            else:
                filter_flag = (example1[self.variable_of_interest_col_name][0] >= 0)
            if filter_flag:
                features.append(example1)
                labels.append(example1[self.variable_of_interest_col_name])
                # print(self.variable_of_interest_col_name, example1[self.variable_of_interest_col_name])
                # if "Subject" in example1.keys():
                #     id_name = "Subject"
                # else:
                #     id_name = "subject"
                # idx = example1[id_name][0] if "Shef" in self.struct_features_path else int(example1["Filename"].split(".")[0].split("-")[-1])
                idx = int(example1["Filename"].split(".")[0].split("-")[-1])
                if self.structure_features is not None:
                    struc_features.append(self.structure_features[self.structure_features["SubjectID"] == idx].values[0, 1:])
            # Wrap each value in the key:value pairs into a list (expected by preprocess() and collate())
            
        return features, labels, struc_features

    def __getitem__(self, idx):
        example = self.features[idx]
        

        processed_example = preprocess_fmri(example, self.coords_ds, self.recording_col_name, variable_of_interest_col_name=self.variable_of_interest_col_name, moving_window_len=self.moving_window_len)
        # label = (torch.round(processed_example["label"]/20) - 1).long()
        if "vas" in self.variable_of_interest_col_name.lower():
            label = (processed_example["label"]//20 - 1).long()
            if label == 4:
                label = torch.tensor(3, dtype=torch.int64)  # Map 4 to 3
        else:
            label = processed_example["label"]
        # print("After preprocess:", processed_example["brain_network"].shape)

        return {
            "signal_vectors": processed_example["signal_vectors"].squeeze(0),
            "xyz_vectors": processed_example["xyz_vectors"].squeeze(0),
            "input_ids": processed_example["signal_vectors"].squeeze(0),
            # "labels": processed_example["label"].squeeze(),   
            "labels": label,
            "idx": torch.tensor(idx, dtype=torch.int64),
            # "brain_network": processed_example["brain_network"],
            "structure_feature": self.structure_features[idx] if self.structure_features is not None else None
        }
    


def collate_fn(example):
    """
    This function tells the dataloader how to stack a batch of examples from the dataset.
    Need to stack gene expression vectors and maintain same argument names for model inputs
    which CellLM is expecting in forward() function:
        expression_vectors, sampled_gene_indices, and cell_indices
    """
    # These inputs will go to model.forward(), names must match
    return {
        "signal_vectors": torch.stack([e["signal_vectors"] for e in example]),
        "xyz_vectors": torch.stack([e["xyz_vectors"] for e in example]),
        "input_ids": torch.stack([e["signal_vectors"] for e in example]),
        "labels": torch.stack([e["labels"] for e in example]),
        "structure_feature": torch.stack([e["structure_feature"] for e in example]) if example[0]["structure_feature"] is not None else None,
        "idx": torch.stack([e["idx"] for e in example]),
    }