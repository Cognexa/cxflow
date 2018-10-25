"""
Test module for config utils functions (cxflow.utils.config).
"""
from os import path
from collections import OrderedDict
import pytest

from cxflow.utils.config import parse_arg, load_config


_ARGS = [([('common.name', 'BatchSize1'), ('model.name', 'modelie'), ('stream.train.seed', 'none')], str),
         ([('common.batch_size', 12), ('model.dropout', 0), ('stream.train.seed', 123)], int),
         ([('common.some_int_number', 12.), ('model.dropout', 0.5), ('stream.train.float_seed', 123.456)], float),
         ([('common.quiet', True), ('model.dropout', False), ('stream.train.float_seed', True)], bool)]


@pytest.mark.parametrize('args, arg_type', _ARGS)
def test_arg_type(args, arg_type):
    """Test case for parse_arg function."""
    for key, val in args:
        parsed_key, parsed_val = parse_arg(key + '=' + str(val))
        assert (key, val) == (parsed_key, parsed_val)
        assert type(parsed_val) == arg_type


def test_ast_type():
    """Test ast type."""
    args = [('common.arch', [1, 2, 3.4, 5]), ('model.arch', {"a": "b"}),
                     ('stream.train.deep', {"a": {"b": ["c", "d", "e"]}}), ('model.arch', 12), ('model.arch', 12.2)]
    for key, val in args:
        parsed_key, parsed_val = parse_arg(key+'='+str(val))
        assert (key, val) == (parsed_key, parsed_val)
        assert type(parsed_val) == type(val)


_ANCHORLESS_KEYS = [([[]], {'e': {'f': 'f', 'h': ['j', 'k']}}),
                    ([['e.f=12']], {'e': {'f': 12, 'h': ['j', 'k']}}),
                    ([['e.x=12']], {'e': {'f': 'f', 'h': ['j', 'k'], 'x': 12}})]


@pytest.mark.parametrize('params, expected_output', _ANCHORLESS_KEYS)
def test_load_anchorless_config(tmpdir, anchorless_yaml, params, expected_output):
    """Test loading of a config without yaml anchors."""

    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchorless_yaml)

    assert load_config(f_name, *params) == expected_output


_ANCHORED_KEYS = [([[]], {'b': 'c', 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 'c'), ('d', 11)]),
                   ([['a.b=12']], {'b': 12, 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 12), ('d', 11)]),
                   ([['e.b=19']], {'b': 'c', 'd': 11}, [('f', 'f'), ('h', ['j', 'k']), ('b', 19), ('d', 11)])]


@pytest.mark.parametrize('params, a_key, e_key', _ANCHORED_KEYS)
def test_load_anchored_config(tmpdir, anchored_yaml, params, a_key, e_key):
    """Test loading of a config with yaml anchors."""
    f_name = path.join(tmpdir, 'conf.yaml')

    with open(f_name, 'w') as file:
        file.write(anchored_yaml)

    assert load_config(f_name, *params)['a'] == a_key
    assert OrderedDict(load_config(f_name, *params)['e']) == OrderedDict(e_key)
