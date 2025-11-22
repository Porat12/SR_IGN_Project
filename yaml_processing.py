import yaml

from models.block_type import BlockType


def block_type_constructor(loader, node):
    value = loader.construct_scalar(node)
    return BlockType[value.upper()]

def register_yaml_constructors():
    """Registers all necessary custom constructors for PyYAML."""
    yaml.add_constructor('!BlockType', block_type_constructor, Loader=yaml.SafeLoader)
    print("Custom YAML constructor for !BlockType registered.")

def read_yaml_config(file_path):
    """Reads a YAML configuration file and returns the parsed content."""
    with open(file_path, 'r') as file:
        config = config = yaml.safe_load(file)
    return config