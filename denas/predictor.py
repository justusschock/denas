from delira.training import Predictor
from delira.training.train_utils import convert_torch_tensor_to_npy


class ENASPredictor(Predictor):
    def __init__(self, model, key_mapping, pred_mode: str,
                 convert_batch_to_npy_fn=convert_torch_tensor_to_npy,
                 prepare_batch_fn=lambda x: x, **kwargs):

        self.pred_mode = pred_mode

        super().__init__(model, key_mapping, convert_batch_to_npy_fn,
                         prepare_batch_fn, **kwargs)

    def predict(self, data):
        data = self._prepare_batch(data)

        mapped_data = {
            k: data[v] for k, v in self.key_mapping.items()}

        pred = self.module(
            self.pred_mode, **mapped_data
        )

        # converts positional arguments and keyword arguments,
        # but returns only keyword arguments, since positional
        # arguments are not given.
        return self._convert_batch_to_npy_fn(
            **pred
        )[1]
