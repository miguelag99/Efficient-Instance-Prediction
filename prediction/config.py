import argparse
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

