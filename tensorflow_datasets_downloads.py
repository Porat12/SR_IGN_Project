import tensorflow_datasets as tfds

tfds.load("div2k", split="train", data_dir="./data", download=True)