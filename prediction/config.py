import argparse
import importlib
import os

from types import SimpleNamespace

def namespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        result = {}
        for key, value in namespace.__dict__.items():
            result[key] = namespace_to_dict(value)
        return result
    elif isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    else:
        return namespace


def get_file_names(path: str = './prediction/configs/'):
    """Obtain the names of the files in the specified path.

    Args:
        path (str): path with files. Defaults to './prediction/configs/'.
        
    Returns:
        f_names (list): list of file names without the extension.
    """
    
    f_names = os.listdir(path)
    f_names = list(map(lambda x: x.split('.')[0], f_names))
    
    return f_names
    
    
def get_config(cfg_dir: str = './prediction/configs/',
               cfg_file: str = 'b0_short'):
    """Check if the specified configuration file exists in the specified directory
    and return the configuration as a SimpleNamespace object.

    Args:
        cfg_dir (str): Path to config files.
            Defaults to './prediction/configs/'.
        cfg_file (str): name of the config file. Defaults to 'b0_short'.
        
    Returns:
        model_cfg (SimpleNamespace): configuration as a SimpleNamespace object.
        hparams (dict): configuration as a dictionary.
    """
    
    cfg_files = get_file_names(cfg_dir)
    
    assert cfg_file in cfg_files, f"Config file {cfg_file} not found in {cfg_dir}"

    lib_comps = [dir for dir in cfg_dir.split(os.sep) if dir and dir != '.']
    lib_comps.append(cfg_file)
    
    module_name = '.'.join(lib_comps)
    config_module = importlib.import_module(module_name)

    model_cfg = config_module.model_cfg
    hparams = namespace_to_dict(model_cfg)
    
    return model_cfg, hparams
    
    