import torch


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
history_keys = ["total_loss", "rec_loss", "idem_loss", "tight_loss", "SR_loss"]
wandb_key = "7d297933fe02b9f35d919bd5072556f8408deb19"  # for HPC