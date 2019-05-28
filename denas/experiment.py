from delira.training import PyTorchExperiment
from .predictor import ENASPredictor
import typing
from delira.training import Parameters
from delira.models import AbstractPyTorchNetwork
from delira.training.callbacks import CosineAnnealingLRCallbackPyTorch
from .trainer import ENASTrainerPyTorch
from .utils import create_optims_enas
from delira.data_loading import BaseDataManager


class ENASExperimentPyTorch(PyTorchExperiment):

    def __init__(self,
                 params: typing.Union[str, Parameters],
                 model_cls: AbstractPyTorchNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=create_optims_enas,
                 checkpoint_freq=10,
                 trainer_cls=ENASTrainerPyTorch,
                 **kwargs):

        super().__init__(params=params, model_cls=model_cls, n_epochs=n_epochs,
                         name=name, save_path=save_path, key_mapping=key_mapping,
                         val_score_key=val_score_key,
                         optim_builder=optim_builder,
                         checkpoint_freq=checkpoint_freq,
                         trainer_cls=trainer_cls, **kwargs)

    def _setup_test(self, params, model, convert_batch_to_npy_fn,
                    prepare_batch_fn, **kwargs):
        predictor = ENASPredictor(model=model, key_mapping=self.key_mapping,
                                  convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                                  prepare_batch_fn=prepare_batch_fn, **kwargs)
        return predictor

    def _setup_training(self, params, T_max, eta_min, **kwargs):
        trainer = super()._setup_training(params, **kwargs)
        callback = CosineAnnealingLRCallbackPyTorch(
            trainer.optimizers["shared_cnn"], T_max=T_max, eta_min=eta_min)
        trainer.register_callback(callback)
        return trainer

    def run(self, train_data_controller: BaseDataManager,
            train_data_shared_cnn: BaseDataManager,
            val_data: BaseDataManager = None,
            params: Parameters = None, **kwargs):

        """
        Setup and run training

        Parameters
        ----------
        train_data : :class:`BaseDataManager`
            the data to use for training
        val_data : :class:`BaseDataManager` or None
            the data to use for validation (no validation is done
            if passing None); default: None
        params : :class:`Parameters` or None
            the parameters to use for training and model instantiation
            (will be merged with ``self.params``)
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            The trained network returned by the trainer (usually best network)

        See Also
        --------
        :class:`BaseNetworkTrainer` for training itself

        """

        params = self._resolve_params(params)
        kwargs = self._resolve_kwargs(kwargs)

        params.permute_training_on_top()
        training_params = params.training

        trainer = self.setup(params, training=True, **kwargs)

        self._run += 1

        num_epochs = kwargs.get("num_epochs", training_params.nested_get(
            "num_epochs", self.n_epochs))

        if num_epochs is None:
            num_epochs = self.n_epochs

        return trainer.train(num_epochs, train_data_controller,
                             train_data_shared_cnn, val_data,
                             self.val_score_key, kwargs.get("val_score_mode",
                                                            "lowest"))