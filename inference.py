import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_1mDFDC_train,
                        Dataset_1mDFDC_devNeval, genSpoof_list)
from evaluation_1m import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from tqdm import tqdm
from main_1m import produce_evaluation_file, get_model, get_custom_loader

def main(args: argparse.Namespace) -> None:
    try:
        print("Reading configuration file...")
        with open(args.config, "r") as f_json:
            config = json.loads(f_json.read())
        print("Configuration loaded.")

        model_config = config["model_config"]
        optim_config = config["optim_config"]
        optim_config["epochs"] = config["num_epochs"]
        track = config["track"]
        assert track in ["LA", "PA", "DF"], "Invalid track given"
        print("Track validated.")

        database_path = Path(config["database_path"]) # Path to the 1mDFDC database
        # define model related paths
        model_tag = "{}_{}_ep{}_bs{}".format(
            track,
            os.path.splitext(os.path.basename(args.config))[0],
            config["num_epochs"], config["batch_size"])
        model_tag = Path(args.output_dir) / model_tag
        metric_path = model_tag / "metrics"
    
        ckpt_dir = Path('/media/NAS/USERS/moonbo/aasist/weights/exp_result/DF_AASIST_1mdfdc_ep50_bs64/')
        ckpt_file = 'epoch_2_2.136.pth'
        ckpt = ckpt_dir / ckpt_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        model = get_model(model_config, device)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        print("Device: {}".format(device))

        print("Loading model checkpoint...")
        if torch.cuda.is_available:
            model.load_state_dict(torch.load(ckpt), strict=False)
        
        # make file list
        print("Generating file lists...")
        train_files = []
        val_files = []
        test_files = []
        file_folder = '/media/NAS/DATASET/1mDFDC/filelist/'
        train_file_path = os.path.join(file_folder, 'train_audio.txt')
        val_file_path = os.path.join(file_folder, 'val_audio.txt')
        test_file_path = os.path.join(file_folder, 'test_audio.txt')

        eval_trial_path = test_file_path
        eval_score_path = model_tag / config["eval_output"]

        with open(train_file_path, 'r') as f:
            train_files = f.readlines()
        with open(val_file_path, 'r') as f:
            val_files = f.readlines()
        with open(test_file_path, 'r') as f:
            test_files = f.readlines()
        print("File lists generated.")

        print("Creating data loaders...")
        trn_loader, dev_loader, eval_loader = get_custom_loader(train_files, val_files, test_files, args.seed, config)
        print("Data loaders created.")

        print("Producing evaluation file...")
        produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path, eval=True)
        print("Evaluation file produced.")
        
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path / "t-DCF_EER_{%s}epo.txt".format(model_tag)
        )

        print("EER and t-DCF calculated.")
        f_log = open(model_tag / "metric_log_eval.txt", "a")
        f_log.write("=" * 5 + "\n")
        f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
        f_log.close()
        print("Evaluation completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST Inference Script")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./inference",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    args = parser.parse_args()
    main(args)
