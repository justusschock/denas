"""
DENAS - Delira Efficient Neural Network Search
"""

from .models import SeparableConv, PoolBranch, FixedLayer, FactorizedReduction, \
    ENASLayer, ConvBranch, SharedCNN, Controller

from .train import evaluate_model, main, get_best_arc, get_eval_accuracy, \
    load_datasets, print_arc, train_controller, train_enas, train_fixed, \
    train_shared_cnn