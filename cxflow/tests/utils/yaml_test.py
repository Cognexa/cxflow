"""
Test module for yaml utils functions (cxflow.utils.yaml).
"""
from os import path
from collections import OrderedDict

import ruamel.yaml
import yaml

from cxflow.utils.yaml import yaml_to_file, yaml_to_str, load_yaml, make_simple
from cxflow.constants import CXF_CONFIG_FILE


def test_load_anchorless_yaml(tmpdir, anchorless_yaml):
    """Test loading of a YAML without yaml anchors."""

    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchorless_yaml)

    assert load_yaml(f_name) == {'e': {'f': 'f', 'h': ['j', 'k']}}


def test_load_anchored_yaml(tmpdir, anchored_yaml):
    """Test loading of a YAML with yaml anchors."""
    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchored_yaml)

    assert load_yaml(f_name)['a'] == {'b': 'c', 'd': 11}
    assert OrderedDict(load_yaml(f_name)['e']) == OrderedDict([('f', 'f'), ('h', ['j', 'k']), ('b', 'c'), ('d', 11)])


def test_dump_yaml(tmpdir):
    """Test yaml_to_file and yaml_to_str function."""

    config = {'e': {'f': 'f', 'h': ['j', 'k']}}

    # test if the return path is correct and re-loading does not change the config
    config_path = yaml_to_file(config, output_dir=tmpdir, name=CXF_CONFIG_FILE)
    assert path.exists(config_path)
    assert load_yaml(config_path) == config

    # test custom naming
    dump_name = 'my-conf.yaml'
    config_path = yaml_to_file(config, output_dir=tmpdir, name=dump_name)
    assert path.join(tmpdir, dump_name) == config_path
    assert path.exists(config_path)

    # test dump to string (effectively, test pyaml)
    yaml_str = yaml_to_str(config)
    assert yaml.load(yaml_str) == config


def test_make_simple(anchored_yaml):
    """Test yaml make_simple utility function."""
    anchored = ruamel.yaml.load(anchored_yaml, Loader=ruamel.yaml.RoundTripLoader)
    full_part = yaml.load(yaml_to_str(make_simple(anchored)['e']))
    assert full_part['b'] == 'c'
