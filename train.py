from denas import ENASExperimentPyTorch, ENASModelPyTorch
from denas.utils import Config, accuracy_metric
from delira.training import Parameters
from delira.data_loading.dataset import TorchvisionClassificationDataset
from delira.data_loading import BaseDataManager
import os

from batchgenerators.transforms import Compose, RandomCropTransform, \
    PadTransform, MirrorTransform, ZeroMeanUnitVarianceTransform

import torch


def create_datasets(config: dict, **kwargs):
    batchsize = config["training"].pop("batchsize")
    num_processes = config["training"].pop("num_processes")

    data_path = config["training"].pop("data_path",
                                       os.path.join(os.getcwd(), "data"))

    dset_train = TorchvisionClassificationDataset(
        "cifar10",
        root=data_path,
        train=True, download=True,
        img_shape=(32, 32), one_hot=False,
        **kwargs)

    dset_val = TorchvisionClassificationDataset(
        "cifar10",
        root=data_path,
        train=False, download=True,
        img_shape=(32, 32), one_hot=False,
        **kwargs)

    train_trafos = Compose([
        PadTransform((36, 36)),
        RandomCropTransform(32),
        MirrorTransform((1,)),
        ZeroMeanUnitVarianceTransform()
    ])

    val_trafos = Compose([
        ZeroMeanUnitVarianceTransform()
    ])

    dmgr_train_controller = BaseDataManager(dset_train, batchsize,
                                            n_process_augmentation=num_processes,
                                            transforms=train_trafos)
    dmgr_train_shared_cnn = BaseDataManager(dset_train, batchsize,
                                            n_process_augmentation=num_processes,
                                            transforms=train_trafos)

    dmgr_val = BaseDataManager(dset_val, batchsize,
                               n_process_augmentation=num_processes,
                               transforms=val_trafos)

    return {"train_controller": dmgr_train_controller,
            "train_shared_cnn": dmgr_train_shared_cnn,
            "val": dmgr_val}


def create_experiment_from_config(config: dict):

    params = Parameters(
        fixed_params={
            "model": {
                "search_for": config["controller"].pop("search_for", "macro"),
                "search_whole_channels": config["controller"].pop(
                    "search_whole_channels", True),
                "child_num_layers": config["child"].pop("num_layers", 12),
                "child_num_branches": config["child"].pop("num_branches", 6),
                "child_out_filters": config["child"].pop("out_filters", 36),
                "controller_lstm_size": config["controller"].pop("lstm_size", 64),
                "controller_lstm_num_layers": config["controller"].pop(
                    "num_layers", 1),
                "controller_tanh_constant": config["controller"].pop(
                    "tanh_constant", 1.5),
                "controller_skip_target": config["controller"].pop(
                    "skip_target", 0.4),
                "controller_skip_weight": config["controller"].pop(
                    "skip_weight", 0.8),
                "child_keep_prob": config["child"].pop("keep_prob", 0.9),
                "controller_num_aggregates": config["controller"].pop(
                    "num_aggregates", 20),
                "controller_baseline_decay": config["controller"].pop(
                    "baseline_decay", 0.99),
                "controller_entropy_weight": config["controller"].pop(
                    "entropy_weight", 0.0001),
                "child_grad_bound": config["child"].pop("grad_bound", 5.0)
            },
            "training": {
                "num_epochs": 500,
                "losses": {
                    "shared_cnn": torch.nn.CrossEntropyLoss()
                },
                "val_metrics": {"acc": accuracy_metric},
                "optimizer_cls": {"controller": torch.optim.Adam,
                                  "shared_cnn": torch.optim.SGD},
                "optimizer_params": {
                    "shared_cnn": {"lr": config["child"].pop("lr_max"),
                                   "momentum": 0.9, "nesterov": True,
                                   "weight_decay": config["child"].pop("l2_reg")},
                    "controller": {
                        "lr": config["controller"].pop("lr"),
                        "betas": (0.0, 0.999),
                        "eps": 1e-3
                    }
                }
            }
        }
    )

    experiment = ENASExperimentPyTorch(params, ENASModelPyTorch,
                                       config["training"].pop("num_epochs", 50),
                                       save_path=config["training"].pop(
                                           "save_path", None),
                                       val_score_key="val_acc"
                                       )

    return experiment


def start_training(config_path: str = "./enas.config", dset_kwargs: dict = {},
                   **kwargs):

    config = Config()(config_path)

    data = create_datasets(config, **dset_kwargs)

    experiment = create_experiment_from_config(config)

    experiment.run(train_data_controller=data["train_controller"],
                   train_data_shared_cnn=data["train_shared_cnn"],
                   val_data=data["val"], T_max=config["child"].pop("T_max"),
                   eta_min=config["child"].pop("lr_min"), **kwargs)


if __name__ == '__main__':
    DSET_KWARGS = {}
    KWARGS = {"gpu_ids": [0]}
    CONFIG_PATH = "./enas.config"
    start_training(CONFIG_PATH, DSET_KWARGS, **KWARGS)