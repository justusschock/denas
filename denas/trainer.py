from delira.training import PyTorchNetworkTrainer
from batchgenerators.dataloading import MultiThreadedAugmenter
from tqdm import tqdm
from .models import ENASModelPyTorch
import torch
import numpy as np
import logging


class ENASTrainerPyTorch(PyTorchNetworkTrainer):

    def _setup(self, network: ENASModelPyTorch, optim_fn, optimizer_cls,
               optimizer_params, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               key_mapping, convert_batch_to_npy_fn, mixed_precision,
               mixed_precision_kwargs):

        self.closure_fn_shared_cnn = network.closure_shared_cnn
        self.closure_fn_controller = network.closure_controller

        super()._setup(network, optim_fn, optimizer_cls, optimizer_params,
                       lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                       key_mapping, convert_batch_to_npy_fn, mixed_precision,
                       mixed_precision_kwargs)

    def train(self, num_epochs, datamgr_train_controller,
              datamgr_train_shared_cnn, datamgr_valid=None,
              val_score_key=None, val_score_mode='highest', reduce_mode='mean',
              verbose=True, n_samples_val=100):
        """
        Defines a routine to train a specified number of epochs

        Parameters
        ----------
        num_epochs : int
            number of epochs to train
        datamgr_train_controller : DataManager
            the datamanager holding the train data for training the controller
        datamgr_train_shared_cnn : DataManager
            the datamanager holding the train data for training the shared_cnn
        datamgr_valid : DataManager
            the datamanager holding the validation data (default: None)
        val_score_key : str
            the key specifying which metric to use for validation
            (default: None)
        val_score_mode : str
            key specifying what kind of validation score is best
        reduce_mode : str
            'mean','sum','first_only'
        verbose : bool
            whether to show progress bars or not
        n_samples_val : int
            Number of samples to predict from for determining best architecture

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        self._at_training_begin()

        if val_score_mode == 'highest':
            best_val_score = 0
        elif val_score_mode == 'lowest':
            best_val_score = float('inf')
        else:
            best_val_score = None

        is_best = False
        new_val_score = best_val_score

        if reduce_mode == 'mean':
            def reduce_fn(batch):
                return np.mean(batch)
        elif reduce_mode == 'sum':
            def reduce_fn(batch):
                return np.sum(batch)
        elif reduce_mode == 'first_only':
            def reduce_fn(batch):
                return batch[0]
        elif reduce_mode == 'last_only':
            def reduce_fn(batch):
                return batch[-1]
        else:
            raise ValueError("No valid reduce mode given")

        metrics_val = {}

        val_metric_fns = {}

        for k, v in self.val_metrics.items():
            if not k.startswith("val_"):
                k = "val_" + k

            val_metric_fns[k] = v

        if self.metric_keys is None:
            val_metric_keys = None

        else:
            val_metric_keys = {}
            for k, v in self.metric_keys.items():
                if not k.startswith("val_"):
                    k = "val_" + k

                val_metric_keys[k] = v

        for epoch in range(self.start_epoch, num_epochs + 1):

            self._at_epoch_begin(metrics_val, val_score_key, epoch,
                                 num_epochs)

            batch_gen_train_shared_cnn = datamgr_train_shared_cnn.get_batchgen(
                seed=epoch)
            batchgen_train_controller = datamgr_train_controller.get_batchgen(
                seed=epoch
            )

            # train single network epoch
            train_metrics, train_losses = self._train_single_epoch(
                batch_gen_train_shared_cnn, batchgen_train_controller, epoch,
                verbose=verbose)

            total_metrics = {
                **train_metrics,
                **train_losses}

            if datamgr_valid is not None:
                preds_val, metrics_val = self._evaluate_single_epoch(
                    datamgr_valid,
                    datamgr_train_controller,
                    metrics=self.val_metrics,
                    metric_keys=self.metric_keys,
                    verbose=verbose,
                    epoch=epoch)

            total_metrics.update(metrics_val)

            for k, v in total_metrics.items():
                total_metrics[k] = reduce_fn(v)

            # check if metric became better
            if val_score_key is not None:
                if val_score_key not in total_metrics:
                    if "val_" + val_score_key not in total_metrics:
                        logging.warning("val_score_key '%s' not a valid key for \
                                    validation metrics ")

                        new_val_score = best_val_score

                    else:
                        new_val_score = total_metrics["val_" + val_score_key]
                        val_score_key = "val_" + val_score_key
                else:
                    new_val_score = total_metrics.get(val_score_key)

            if new_val_score != best_val_score:
                is_best = self._is_better_val_scores(
                    best_val_score, new_val_score, val_score_mode)

                # set best_val_score to new_val_score if is_best
                best_val_score = int(is_best) * new_val_score + \
                                 (1 - int(is_best)) * best_val_score

                if is_best and verbose:
                    logging.info("New Best Value at Epoch %03d : %03.3f" %
                                 (epoch, best_val_score))

            # log metrics and loss values
            for key, val in total_metrics.items():
                logging.info({"value": {"value": val, "name": key
                                        }})

            self._at_epoch_end(total_metrics, val_score_key, epoch, is_best)

            is_best = False

            # stop training (might be caused by early stopping)
            if self.stop_training:
                break

        return self._at_training_end(datamgr_valid, n_samples_val, verbose=verbose)

    def _at_training_end(self, batchgen: MultiThreadedAugmenter, n_samples: int,
                         verbose: bool):
        self.get_best_arc(batchgen, n_samples=n_samples, verbose=verbose)
        return super()._at_training_end()

    def _train_single_epoch(self,
                            batchgen_train_shared_cnn: MultiThreadedAugmenter,
                            batchgen_train_controller: MultiThreadedAugmenter,
                            epoch: int, verbose=False):

        metrics_shared_cnn, losses_shared_cnn = \
            self._train_single_epoch_shared_cnn(batchgen_train_shared_cnn,
                                                epoch, verbose)

        metrics_controller, losses_controller = \
            self._train_single_epoch_controller(batchgen_train_controller,
                                                epoch, verbose)

        return ({**metrics_shared_cnn, **metrics_controller},
                {**losses_shared_cnn, **losses_controller})

    def _train_single_epoch_shared_cnn(self, batchgen: MultiThreadedAugmenter,
                                       epoch, verbose=False):
        """
        Trains the shared network a single epoch with the generated architecture

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the training batches
        epoch : int
            current epoch

        """

        metrics, losses = [], []

        self.module.controller.eval()
        self.module.shared_cnn.train()

        n_batches = batchgen.generator.num_batches * batchgen.num_processes
        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' batch', total=n_batches,
                            desc='Epoch %d SharedCNN' % epoch)
        else:
            iterable = enumerate(batchgen)

        for batch_nr, batch in iterable:
            data_dict = self._prepare_batch(batch)

            _metrics, _losses, _ = self.closure_fn_shared_cnn(
                self.module,
                data_dict,
                optimizers=self.optimizers,
                losses=self.losses,
                metrics=self.train_metrics,
                fold=self.fold,
                batch_nr=batch_nr)

            metrics.append(_metrics)
            losses.append(_losses)

        batchgen._finish()

        self.module.controller.train()

        total_losses, total_metrics = {}, {}

        for _metrics in metrics:
            for key, val in _metrics.items():
                if key in total_metrics:
                    total_metrics[key].append(val)
                else:
                    total_metrics[key] = [val]

        for _losses in losses:
            for key, val in _losses.items():
                if key in total_losses:
                    total_losses[key].append(val)
                else:
                    total_losses[key] = [val]

        return total_metrics, total_losses

    def _train_single_epoch_controller(self, batchgen: MultiThreadedAugmenter,
                                       epoch, verbose=False):
        """
        Trains the controller network a single epoch

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the training batches
        epoch : int
            current epoch

        """

        metrics, losses = [], []

        self.module.controller.train()
        self.module.shared_cnn.eval()

        n_batches = batchgen.generator.num_batches * batchgen.num_processes
        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' batch', total=n_batches,
                            desc='Epoch %d Controller' % epoch)
        else:
            iterable = enumerate(batchgen)

        for batch_nr, batch in iterable:
            data_dict = self._prepare_batch(batch)

            _metrics, _losses, _ = self.closure_fn_controller(
                self.module,
                data_dict,
                optimizers=self.optimizers,
                losses=self.losses,
                metrics=self.train_metrics,
                fold=self.fold,
                batch_nr=batch_nr)

            metrics.append(_metrics)
            losses.append(_losses)

        batchgen._finish()

        self.module.shared_cnn.train()

        total_losses, total_metrics = {}, {}

        for _metrics in metrics:
            for key, val in _metrics.items():
                if key in total_metrics:
                    total_metrics[key].append(val)
                else:
                    total_metrics[key] = [val]

        for _losses in losses:
            for key, val in _losses.items():
                if key in total_losses:
                    total_losses[key].append(val)
                else:
                    total_losses[key] = [val]

        return total_metrics, total_losses

    def get_best_arc(self, batchgen, n_samples=10, verbose=False):
        """Evaluate several architectures and return the best performing one.

        Args:
            controller: Controller module that generates architectures to be trained.
            shared_cnn: CNN that contains all possible architectures, with shared weights.
            data_loaders: Dict containing data loaders.
            n_samples: Number of architectures to test when looking for the best one.
            verbose: If True, display the architecture and resulting validation accuracy.

        Returns:
            best_arc: The best performing architecture.
            best_vall_acc: Accuracy achieved on the best performing architecture.

        All architectures are evaluated on the same minibatch from the validation set.
        """

        self.module.eval()

        n_batches = batchgen.generator.num_batches * batchgen.num_processes
        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' batch', total=n_batches,
                            desc='Evaluate for best Architecture')
        else:
            iterable = enumerate(batchgen)


        arcs = []
        val_accs = []
        for idx, batch in iterable:
            if idx >= n_samples:
                break
            batch = self._prepare_batch(batch)

            with torch.no_grad():
                sample_arc = self.module("controller")["pred"]  # perform forward pass to generate a new architecture
            arcs.append(sample_arc)

            with torch.no_grad():
                pred = self.module("shared_cnn", batch["data"], sample_arc)
            val_acc = torch.mean((torch.max(pred["pred"], 1)[1] == batch["label"]).float())
            val_accs.append(val_acc.item())

            if verbose:
                self.print_arc(sample_arc)
                print('val_acc=' + str(val_acc.item()))
                print('-' * 80)

        best_iter = int(np.argmax(val_accs))
        best_arc = arcs[best_iter]
        best_val_acc = val_accs[best_iter]

        self.module.train()
        return best_arc, best_val_acc

    @staticmethod
    def print_arc(sample_arc):
        """Display a sample architecture in a readable format.

        Args:
            sample_arc: The architecture to display.

        Returns: Nothing.
        """
        for key, value in sample_arc.items():
            if len(value) == 1:
                branch_type = value[0].cpu().numpy().tolist()
                print('[' + ' '.join(str(n) for n in branch_type) + ']')
            else:
                branch_type = value[0].cpu().numpy().tolist()
                skips = value[1].cpu().numpy().tolist()
                print('[' + ' '.join(str(n) for n in (branch_type + skips)) + ']')

    def predict(self, mode: str, data: dict, **kwargs):

        data = self._prepare_batch(data)

        mapped_data = {
            k: data[v] for k, v in self.key_mapping.items()}

        pred = self.module(
            mode,
            **mapped_data,
            **kwargs
        )

        # converts positional arguments and keyword arguments,
        # but returns only keyword arguments, since positional
        # arguments are not given.
        return self._convert_to_npy_fn(
            **pred
        )[1]

    def _evaluate_single_epoch(self, datamgr_val, dmgr_train_controller,
                               batchsize=None, metrics={},
                               metric_keys=None, verbose=False, epoch=None,
                               n_samples=10):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False

        Returns
        -------
        dict
            a dictionary containing all predictions
        dict
            a dictionary containing all validation metrics (maybe empty)

        """
        self.module.eval()

        if epoch is None:
            seed = 1
        else:
            seed = epoch

        best_arc, _ = self.get_best_arc(
            dmgr_train_controller.get_batchgen(seed=seed),
            n_samples=n_samples, verbose=verbose)

        orig_num_aug_processes = datamgr_val.n_process_augmentation
        orig_batch_size = datamgr_val.batch_size

        if batchsize is None:
            batchsize = orig_batch_size

        datamgr_val.batch_size = 1
        datamgr_val.n_process_augmentation = 1

        batchgen = datamgr_val.get_batchgen()

        predictions_all, metric_vals = [], {k: [] for k in metrics.keys()}

        n_batches = batchgen.generator.num_batches * batchgen.num_processes

        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' sample',
                            total=n_batches, desc=self._tqdm_desc)

        else:
            iterable = enumerate(batchgen)

        batch_list = []

        for i, batch in iterable:

            if not batch_list and (n_batches - i) < batchsize:
                batchsize = n_batches - i
                logging.debug("Set Batchsize down to %d to avoid cutting "
                              "of the last batches" % batchsize)

            batch_list.append(batch)

            # if queue is full process queue:
            if batchsize is None or len(batch_list) >= batchsize:

                batch_dict = {}
                for batch in batch_list:
                    for key, val in batch.items():
                        if key in batch_dict.keys():
                            batch_dict[key].append(val)
                        else:
                            batch_dict[key] = [val]

                for key, val_list in batch_dict.items():
                    batch_dict[key] = np.concatenate(val_list)

                preds = self.predict("shared_cnn", batch_dict,
                                     sample_arc=best_arc)

                # calculate metrics for predicted batch
                _metric_vals = self.calc_metrics({**batch_dict, **preds},
                                                 metrics=metrics,
                                                 metric_keys=metric_keys)

                for k, v in _metric_vals.items():
                    metric_vals[k].append(v)

                predictions_all.append(preds)

                batch_list = []

        batchgen._finish()

        # convert predictions from list of dicts to dict of lists
        new_predictions_all = {}
        for preds in predictions_all:
            for k, v in preds.items():

                # check if v is scalar and convert to npy-array if necessary.
                # Otherwise concatenation might fail
                if np.isscalar(v):
                    v = np.array(v)

                # check for zero-sized arrays and reshape if necessary.
                # Otherwise concatenation might fail
                if v.shape == ():
                    v = v.reshape(1)
                if k in new_predictions_all:
                    new_predictions_all[k].append(v)
                else:
                    new_predictions_all[k] = [v]

        # concatenate lists to single arrays
        predictions_all = {k: np.concatenate(_outputs)
                           for k, _outputs in new_predictions_all.items()}

        for k, v in metric_vals.items():
            metric_vals[k] = np.array(v)

        datamgr_val.batch_size = orig_batch_size
        datamgr_val.n_process_augmentation = orig_num_aug_processes

        self.module.train()
        return predictions_all, metric_vals
