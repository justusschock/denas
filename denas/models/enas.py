from delira.models import AbstractPyTorchNetwork
import torch
from .controller import Controller
from .shared_cnn import SharedCNN


class ENASModelPyTorch(AbstractPyTorchNetwork):
    def __init__(self,
                 search_for="macro",
                 search_whole_channels=True,
                 child_num_layers=12,
                 child_num_branches=6,
                 child_out_filters=36,
                 controller_lstm_size=32,
                 controller_lstm_num_layers=2,
                 controller_tanh_constant=1.5,
                 temperature=None,
                 controller_skip_target=0.4,
                 controller_skip_weight=0.8,
                 child_keep_prob=1.0,
                 child_fixed_arc=None,
                 controller_num_aggregates=20,
                 baseline=None,
                 controller_baseline_decay=0.99,
                 controller_entropy_weight=0.0001,
                 child_grad_bound=5.0
                 ):
        super().__init__()

        self.controller_num_aggregates = controller_num_aggregates
        self.baseline = baseline
        self.controller_baseline_decay = controller_baseline_decay
        self.controller_entropy_weight = controller_entropy_weight
        self.child_grad_bound = child_grad_bound
        self._aggregation_counter = 1

        self._build_model(search_for, search_whole_channels, child_num_layers,
                          child_num_branches, child_out_filters,
                          controller_lstm_size, controller_lstm_num_layers,
                          controller_tanh_constant, temperature,
                          controller_skip_target, controller_skip_weight,
                          child_keep_prob, child_fixed_arc)

    def forward(self, model_name, *args, **kwargs):
        return {"pred": getattr(self, model_name)(*args, **kwargs)}

    def _build_model(self, search_for, search_whole_channels, child_num_layers,
                     child_num_branches, child_out_filters,
                     controller_lstm_size, controller_lstm_num_layers,
                     controller_tanh_constant, temperature,
                     controller_skip_target, controller_skip_weight,
                     child_keep_prob, child_fixed_arc):

        self.controller = Controller(search_for, search_whole_channels,
                                     child_num_layers, child_num_branches,
                                     child_out_filters, controller_lstm_size,
                                     controller_lstm_num_layers,
                                     controller_tanh_constant, temperature,
                                     controller_skip_target,
                                     controller_skip_weight)

        self.shared_cnn = SharedCNN(child_num_layers, child_num_branches,
                                    child_out_filters, child_keep_prob,
                                    child_fixed_arc)

    @property
    def controller_backprop(self):
        if self._aggregation_counter % self.controller_num_aggregates == 0:
            self._aggregation_counter = 1
            return True
        else:
            self._aggregation_counter += 1
            return False

    @property
    def controller_skip_penalties(self):
        return self.controller.skip_penaltys

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        batch["data"] = torch.from_numpy(batch["data"]).to(input_device,
                                                           torch.float)
        batch["label"] = torch.from_numpy(batch["data"]).to(output_device,
                                                            torch.long)

        return batch

    # Closure only defined to fulfill API. Closure has to be split in two
    # separate closures, namely closure_shared_cnn and closure_controller
    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={},
                metrics={}, fold=0, **kwargs):
        raise NotImplementedError

    @staticmethod
    def closure_shared_cnn(model, data_dict: dict, optimizers: dict, losses={},
                           metrics={}, fold=0, **kwargs):

        if isinstance(model, torch.nn.DataParallel):
            child_grad_bound = model.module.child_grad_bound
        else:
            child_grad_bound = model.child_grad_bound

        with torch.no_grad():
            sample_arc = model("controller")

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict.pop("data")
            preds = model("shared_cnn", inputs, sample_arc)

            if data_dict:
                _loss_val = losses["shared_cnn"](preds["pred"],
                                                 data_dict["label"])

                loss_vals["shared_cnn"] = _loss_val.item()
                total_loss += _loss_val

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        metric_vals["sharedcnn_" + key] = metric_fn(
                            preds["preds"], data_dict["label"]).item()

        if optimizers:
            optimizers['shared_cnn'].zero_grad()
            # perform loss scaling via apex if half precision is enabled
            with optimizers["shared_cnn"].scale_loss(total_loss) as scaled_loss:
                scaled_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(),
                                                      child_grad_bound)
            optimizers['shared_cnn'].step()

        else:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        return metric_vals, loss_vals, preds

    @staticmethod
    def closure_controller(model, data_dict: dict, optimizers: dict, losses={},
                           metrics={}, fold=0, **kwargs):

        sample_arc = model("controller")

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        with torch.no_grad():

            inputs = data_dict.pop("data")
            preds = model("shared_cnn", inputs, sample_arc)

            acc = torch.mean((torch.amax(preds["pred"], 1) == data_dict["label"]
                              ).to(torch.float))

        reward = acc.detach()

        if isinstance(model, torch.nn.DataParallel):
            controller_backprop = model.module.controller_backprop
            num_aggregates = model.module.num_aggregates
            controller_baseline_decay = model.module.controller_baseline_decay
            baseline = model.module.baseline
            controller_entropy_weight = model.module.controller_entropy_weight
            sample_entropy = model.module.controller.sample_entropy
            sample_log_prob = model.module.controller.sample_log_prob
            controller_skip_weight = model.module.controller_skip_weight
            controller_skip_penalties = model.module.controller_skip_penalies
            child_grad_bound = model.module.child_grad_bound
        else:
            controller_backprop = model.controller_backprop
            num_aggregates = model.num_aggregates
            baseline = model.baseline
            controller_baseline_decay = model.controller_baseline_decay
            controller_entropy_weight = model.controller_entropy_weight
            sample_entropy = model.controller.sample_entropy
            sample_log_prob = model.controller.sample_log_prob
            controller_skip_weight = model.controller_skip_weight
            controller_skip_penalties = model.controller_skip_penalies
            child_grad_bound = model.child_grad_bound

        reward += controller_entropy_weight * sample_entropy

        if baseline is None:
            baseline = acc
        else:
            baseline -= (1 - controller_baseline_decay) * (baseline - reward)
            baseline = baseline.detach()

        loss = -1 * sample_log_prob * (reward - baseline)

        if controller_skip_weight is not None:
            loss += controller_skip_weight * controller_skip_penalties

        loss = loss / num_aggregates

        with torch.no_grad():
            for key, metric_fn in metrics.items():
                metric_vals["controller_" + key] = metric_fn(
                    preds["pred"], data_dict["label"]).item()

        with optimizers["controller"].scale_loss(loss) as scaled_loss:
            scaled_loss.backward(retain_graph=True)

        if controller_backprop:
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(),
                                                      child_grad_bound)
            optimizers["controller"].step()
            optimizers["controller"].zero_grad()

        if not optimizers:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        if isinstance(model, torch.nn.DataParallel):
            model.module.baseline = baseline
        else:
            model.baseline = baseline

        return metric_vals, loss_vals, preds
