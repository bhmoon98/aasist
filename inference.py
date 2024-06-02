import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from main_1m import produce_evaluation_file_inference, get_model, get_custom_loader

def main(args: argparse.Namespace) -> None:
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
    
    ckpt_dir = Path('/media/NAS/USERS/moonbo/aasist/weights/exp_result/DF_AASIST_1mdfdc_ep50_bs64/')
    ckpt_file = 'epoch_21_1.572.pth'
    ckpt = ckpt_dir / ckpt_file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = get_model(model_config, device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2])
        print(model.device_ids)
        model.to(f"cuda:{model.device_ids[0]}")
    else:
        model.to(device)
    print("Device: {}".format(device))

    print("Loading model checkpoint...")
    if torch.cuda.is_available():
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
    eval_score_path = "0602.txt"

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
    produce_evaluation_file_inference(eval_loader, model, device, eval_score_path, eval_trial_path)
    print("Evaluation file produced.")
    
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST Inference Script")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    args = parser.parse_args()
    main(args)