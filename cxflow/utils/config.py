"""
Config module providing util functions for handling YAML configurations.
"""
import typing

import yaml

from .yaml import load_yaml, reload


def parse_arg(arg: str) -> typing.Tuple[str, typing.Any]:
    """
    Parse CLI argument in format ``key=value`` to ``(key, value)``

    :param arg: CLI argument string
    :return: tuple (key, value)
    :raise: yaml.ParserError: on yaml parse error
    """
    assert '=' in arg, 'Unrecognized argument `{}`. [name]=[value] expected.'.format(arg)

    key = arg[:arg.index('=')]
    value = yaml.load(arg[arg.index('=') + 1:])

    return key, value


def load_config(config_file: str, additional_args: typing.Iterable[str]=()) -> dict:
    """
    Load config from YAML ``config_file`` and extend/override it with the given ``additional_args``.

    :param config_file: path the YAML config file to be loaded
    :param additional_args: additional args which may extend or override the config loaded from the file.
    :return: configuration as dict
    """

    config = load_yaml(config_file)

    for key_full, value in [parse_arg(arg) for arg in additional_args]:
        key_split = key_full.split('.')
        key_prefix = key_split[:-1]
        key = key_split[-1]

        conf = config
        for key_part in key_prefix:
            conf = conf[key_part]
        conf[key] = value

    return reload(config)


__all__ = []
