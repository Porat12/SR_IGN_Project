import torch.optim as optim

def build_optimizer(config, model):
    optimizer_config = config["optimizer"]
    cls = getattr(optim, optimizer_config["name"])
    return cls(model.parameters(), **optimizer_config["params"])


def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    cls = getattr(optim.lr_scheduler, scheduler_config["name"])
    return cls(optimizer, **scheduler_config["params"])
