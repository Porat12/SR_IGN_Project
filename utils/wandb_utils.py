

def log_with_prefix(prefix, d):
    return {f"{prefix}/{k}": v for k, v in d.items()}