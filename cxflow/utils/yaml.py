"""
YAML module providing util functions for handling YAMLs.
"""
from os import path
from typing import Mapping, Any

import ruamel.yaml  # pylint: disable=import-error
import yaml


def load_yaml(yaml_file: str) -> Any:
    """
    Load YAML from file.

    :param yaml_file: path to YAML file
    :return: content of the YAML as dict/list
    """
    with open(yaml_file, 'r') as file:
        return ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)


def yaml_to_file(data: Mapping, output_dir: str, name: str) -> str:
    """
    Save the given object to the given path in YAML.

    :param data: dict/list to be dumped
    :param output_dir: target output directory
    :param name: target filename
    :return: target path
    """
    dumped_config_f = path.join(output_dir, name)
    with open(dumped_config_f, 'w') as file:
        yaml.dump(data, file, Dumper=ruamel.yaml.RoundTripDumper)
    return dumped_config_f


def yaml_to_str(data: Mapping) -> str:
    """
    Return the given given config as YAML str.

    :param data: configuration dict
    :return: given configuration as yaml str
    """
    return yaml.dump(data, Dumper=ruamel.yaml.RoundTripDumper)


def make_simple(data: Any) -> Any:
    """
    Substitute all the references in the given data (typically a mapping or sequence) with the actual values.
    This is useful, if you loaded a yaml with RoundTripLoader and you need to dump part of it safely.

    :param data: data to be made simple (dict instead of CommentedMap etc.)
    :return: simplified data
    """
    return yaml.load(yaml.dump(data, Dumper=ruamel.yaml.RoundTripDumper), Loader=ruamel.yaml.Loader)


def reload(data: Any) -> Any:
    """
    Dump and load yaml data.
    This is useful to avoid many anchor parsing bugs. When you edit a yaml config, reload it to make sure
    the changes are propagated to anchor expansions.

    :param data: data to be reloaded
    :return: reloaded data
    """
    return yaml.load(yaml.dump(data, Dumper=ruamel.yaml.RoundTripDumper), Loader=ruamel.yaml.RoundTripLoader)

__all__ = []
