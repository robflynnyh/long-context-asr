from omegaconf import OmegaConf

def multiply(a, b):
    return a * b

def integer(value):
    return int(value)

OmegaConf.register_new_resolver("multiply", multiply)
OmegaConf.register_new_resolver("integer", integer)