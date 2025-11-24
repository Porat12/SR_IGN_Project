from .loss_registry import LOSS_REGISTRY
from losses import loss_functions # DO NOT erase this import
# Side-effect import: decorators populate LOSS_REGISTRY.

def build_loss(config):
    loss_config = config["loss"]
    train_loss_name = loss_config["train_loss_name"]
    test_loss_name = loss_config["test_loss_name"]

    if train_loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {train_loss_name}")
    
    if test_loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {test_loss_name}")
    
    return LOSS_REGISTRY[train_loss_name], LOSS_REGISTRY[test_loss_name]
