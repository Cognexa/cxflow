"""
Test module for **cxflow** cli utils (cli/util.py).
"""

from cxflow.cli.util import validate_config

from cxflow.utils.config import load_config
from cxflow.utils.yaml import yaml_to_file
import pytest


def test_train_load_config(tmpdir):
    """Test correct config loading."""

    # test a config call with both dataset and model
    good_config = {'dataset': None, 'model': None}
    config_path = yaml_to_file(good_config, tmpdir, 'config.yaml')

    # test return value
    returned_config = load_config(config_path, [])
    validate_config(returned_config)
    assert returned_config == load_config(config_path, [])
    assert returned_config == good_config


_MISSING_CONFIG = [{'dataset': None}, {'model': None}]


@pytest.mark.parametrize('missing_config', _MISSING_CONFIG)
def test_train_load_missing_config(tmpdir, missing_config):
    """Test correct config loading."""

    # test assertion when config is incomplete
    missing_model_config = missing_config
    config_path = yaml_to_file(missing_model_config, tmpdir, 'config.yaml')
    loaded_config = load_config(config_path, [])
    with pytest.raises(AssertionError):
        validate_config(loaded_config)
