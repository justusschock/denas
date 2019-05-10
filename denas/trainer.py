from delira.training import PyTorchNetworkTrainer
from batchgenerators.dataloading import MultiThreadedAugmenter
from tqdm import tqdm


class ENASTrainerPyTorch(PyTorchNetworkTrainer):
    def _train_single_epoch(self, batchgen: MultiThreadedAugmenter, epoch,
                            verbose=False):

        pass

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
                            desc='Epoch %d SharedCNN' % epoch)
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
