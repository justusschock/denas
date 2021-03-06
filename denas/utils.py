from .models import ENASModelPyTorch
import yaml
import numpy as np
import torch


def create_optims_enas(model: ENASModelPyTorch, optim_cls: dict,
                       **optim_params):
    return {
        "controller": optim_cls["controller"](
            model.controller.parameters(), **optim_params["controller"]),
        "shared_cnn": optim_cls["shared_cnn"](
            model.shared_cnn.parameters(), **optim_params["shared_cnn"])}


def accuracy_metric(preds: np.ndarray, labels: np.ndarray):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().detach().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    preds = np.argmax(preds, -1)

    return (preds == labels).astype(np.float).mean()


# taken from https://github.com/justusschock/shapenet/blob/master/shapenet/utils/load_config_file.py

class Config(object):
    """
    Implements parser for configuration files
    """

    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool
            verbosity
        """
        self.verbose = verbose

    def __call__(self, config_file, config_group=None):
        """
        Actual parsing
        Parameters
        ----------
        config_file : string
            path to YAML file with configuration
        config_group : string or None
            group key to return
            if None: return dict of all keys
            if string: return only values of specified group
        Returns
        -------
        dict
            configuration dict

        """
        state_dict = {}

        # open config file
        with open(config_file, 'r') as file:
            docs = yaml.load_all(file)

            # iterate over document
            for doc in docs:

                # iterate over groups
                for group, group_dict in doc.items():
                    for key, vals in group_dict.items():

                        # set attributes with value 'None' to None
                        if vals == 'None':
                            group_dict[key] = None

                    state_dict[group] = group_dict

                    if self.verbose:
                        print("LOADED_CONFIG: \n%s\n%s:\n%s\n%s" % (
                            "=" * 20, str(group), "-" * 20, str(group_dict)))

        if config_group is not None:
            return state_dict[config_group]
        else:
            return state_dict
