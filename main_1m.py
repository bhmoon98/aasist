"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
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

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "1mDFDC.{}".format(track)
    database_path = Path(config["database_path"]) # Path to the 1mDFDC database
    

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = "/media/NAS/USERS/moonbo/aasist/weights/{}".format(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    copy(args.config, model_tag / "config.conf")

    # # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device: {}".format(device))
    # if device == "cpu":
    #     raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # Parallelize model across multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model.to(device)
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # make file list
    train_files = []
    val_files = []
    test_files = []
    file_folder = '/media/NAS/DATASET/1mDFDC/filelist/'
    train_file_path = os.path.join(file_folder, 'train_audio.txt')
    val_file_path = os.path.join(file_folder, 'val_audio.txt')
    test_file_path = os.path.join(file_folder, 'test_audio.txt')
    
    dev_trial_path = val_file_path
    eval_trial_path = test_file_path

    with open(train_file_path, 'r') as f:
        train_files = f.readlines()
    with open(val_file_path, 'r') as f:
        val_files = f.readlines()
    with open(test_file_path, 'r') as f:
        test_files = f.readlines()
    
    trn_loader, dev_loader, eval_loader = get_custom_loader(train_files, val_files, test_files, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    
    # Training
    for epoch in tqdm(range(config["num_epochs"]), desc="Epochs", total=config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        dev_score_format = "dev_score_{}.txt".format(epoch)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/dev_score_format, dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/dev_score_format,
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            save_path = os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
            # 모델을 save_path에 저장합니다.
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(),
            #            model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # # do evaluation whenever best model is renewed
            # if str_to_bool(config["eval_all_best"]):
            #     produce_evaluation_file(eval_loader, model, device,
            #                             eval_score_path, eval_trial_path)
            #     eval_eer, eval_tdcf = calculate_tDCF_EER(
            #         cm_scores_file=eval_score_path,
            #         asv_score_file=database_path / config["asv_score_path"],
            #         output_file=metric_path /   
            #         "t-DCF_EER_{:03d}epo.txt".format(epoch))

            #     log_text = "epoch{:03d}, ".format(epoch)
            #     if eval_eer < best_eval_eer:
            #         log_text += "best eer, {:.4f}%".format(eval_eer)
            #         best_eval_eer = eval_eer
            #     if eval_tdcf < best_eval_tdcf:
            #         log_text += "best tdcf, {:.4f}".format(eval_tdcf)
            #         best_eval_tdcf = eval_tdcf
            #         torch.save(model.state_dict(),
            #                    model_save_path / "best.pth")
            #     if len(log_text) > 0:
            #         print(log_text)
            #         f_log.write(log_text + "\n")

            # print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path, eval=True)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    save_path = os.path.join(model_save_path, "swa.pth")
    torch.save(model.state_dict(), save_path)
    # torch.save(model.state_dict(),
    #            model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        save_path = os.path.join(model_save_path, "best.pth")
        torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(),
        #            model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

from torch.utils.data import DataLoader
import torch

def get_custom_loader(
        train_files: list,
        val_files: list,
        test_files: list,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    
    train_set = Dataset_1mDFDC_train(list_IDs=train_files)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    dev_set = Dataset_1mDFDC_devNeval(list_IDs=val_files)

    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    eval_set = Dataset_1mDFDC_devNeval(list_IDs=test_files)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    eval: bool=False) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    
    for batch_x, utt_id in tqdm(data_loader, desc="Evaluation Batches", leave=False, total=len(data_loader)):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    if not eval:
        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                fname = os.path.splitext(os.path.basename(trl))[0]
                assert fn == fname
                src = 1 if "real" in fname else 0
                key = src
                fh.write("{} {} {} {}\n".format(fn, src, key, sco))
    else:
        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                fname = os.path.basename(trl) # 확장자 까지 표시
                fh.write("{};{}\n".format(fn, sco))
    print("Scores saved to {}".format(save_path))

def produce_evaluation_file_inference(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list0 = []
    score_list1 = []
    
    for batch_x, utt_id in tqdm(data_loader, desc="Evaluation Batches", leave=False, total=len(data_loader)):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score0 = (batch_out[:, 0]).data.cpu().numpy().ravel()
            batch_score1 = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list0.extend(batch_score0.tolist())
        score_list1.extend(batch_score1.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list0) == len(score_list1)
    with open(save_path, "w") as fh:
        for fn, sco0, sco1 in zip(fname_list, score_list0, score_list1):
            fh.write("{} {} {}\n".format(fn, sco0, sco1))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader, desc="Training Batches", leave=False, total=len(trn_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
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
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
