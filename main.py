import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_from_disk, concatenate_datasets


import random
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold



from dataset.dataset import ArrowDataset, collate_fn
from model.model import MultimodalfMRI

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving_window_len", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel_structure_features", type=int, default=-1)
    parser.add_argument("--label_name", type=str, default="Response")
    parser.add_argument("--brain_lm_path", type=str, default="./brainlm_mae/pretrained_models/2023-06-06-22_15_00-checkpoint-1400")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    moving_window_len = args.moving_window_len
    kfold = args.kfold

    data_path = "/media/wenrui/Data/NeuroPain/fMRI/OpenNeuro/Arrow_patient/"
    structure_features_path = "/media/wenrui/Data/NeuroPain/fMRI/OpenNeuro/struct_stats.csv"


    output_path = f"./results"
    
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H_%M_%S")
    output_path = os.path.join(output_path, f"seed_{args.seed}-{dt_string}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_ds = load_from_disk(os.path.join(data_path, "data"))
    coords_ds = load_from_disk(os.path.join(data_path, "Brain_Region_Coordinates"))
    concat_ds = concatenate_datasets([data_ds])
    

    variable_of_interest_col_name = args.label_name
    col_name = "Raw_Recording"
    """
    Choose from the following normalization methods provided in BrainLM data preparation:
    [
        "Raw_Recording",
        "Subtract_Mean_Divide_Global_STD_Normalized_Recording",
        "All_Patient_All_Voxel_Normalized_Recording",
        "Per_Patient_All_Voxel_Normalized_Recording",
        "Per_Patient_Per_Voxel_Normalized_Recording",
        "Per_Voxel_All_Patient_Normalized_Recording",
        "Subtract_Mean_Normalized_Recording",
    ]
    """

    dataset = ArrowDataset(
        concat_ds, 
        coords_ds, 
        col_name, 
        variable_of_interest_col_name=variable_of_interest_col_name, 
        moving_window_len=moving_window_len, 
        structure_features=structure_features_path
    )

    
    for i, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=kfold, shuffle=False).split(dataset.features, dataset.labels)):

        model_resnet = MultimodalfMRI(
            brain_lm_path=args.brain_lm_path,
            channel_structure_features=args.channel_structure_features,
        ).to(device)
        optimizer = torch.optim.AdamW(model_resnet.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
        

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        valloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        best_mcc = 0
        best_epoch = 0
        loop = tqdm(range(100))
        for epoch in loop:
            loop.set_description(f"Fold: {i} Epoch {epoch}")
            metrics = []
            metrics.append(i)
            optimizer.zero_grad()
            model_resnet.train()
            for example in trainloader:
                logits = model_resnet(example)
                loss = nn.CrossEntropyLoss()(logits, example["labels"].to(device))
                loss.backward()
                optimizer.step()

            model_resnet.eval()
            losses = []
            logits_list = []
            labels_list = []
            for example in valloader:
                with torch.no_grad():
                    logits = model_resnet(example)
                    loss = nn.CrossEntropyLoss()(logits, example["labels"].to(device))
                    losses.append(loss.item())
                    logits_list.append(logits.detach().cpu().numpy())
                    labels_list.append(example["labels"].detach().cpu().numpy())
            logits_list = np.concatenate(logits_list, axis=0)
            labels_list = np.concatenate(labels_list)
            preds_list = np.argmax(logits_list, axis=1)
            acc = accuracy_score(labels_list, preds_list)
            bacc = balanced_accuracy_score(labels_list, preds_list)
            f1 = f1_score(labels_list, preds_list, average='macro')  # or 'micro', 'weighted'
            roc_auc = roc_auc_score(labels_list, logits_list[:, 1])
            mcc = matthews_corrcoef(labels_list, preds_list)

            loop.set_postfix_str(f"Best MCC: {best_mcc:.4f}")
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_epoch = epoch
                save_dict = {
                    "epoch": epoch,
                    "model_resnet": model_resnet.state_dict(),
                    "mcc": best_mcc,
                    "acc": acc,
                    "bacc": bacc,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "args": args,
                    "logits": logits_list,
                    "labels": labels_list
                }
                torch.save(save_dict, os.path.join(output_path, f"Fold_{i}_best_mcc.pt"))

            scheduler.step()