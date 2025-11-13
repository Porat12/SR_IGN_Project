import torch


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
history_keys = ["total_loss", "rec_loss", "idem_loss", "tight_loss", "SR_loss"]