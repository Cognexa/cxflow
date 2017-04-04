"""
Test module for cxflow entry point (entry_point.py)
"""
import logging
import os
from os import path

import yaml

from cxflow.entry_point import train_create_output_dir, train_create_dataset, train_load_config, train_create_hooks
from cxflow.utils.config import config_to_file, load_config
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow.tests.test_core import CXTestCaseWithDir


class DummyDataset:  # pylint: disable=too-few-public-methods
    """Dummy dataset which loads the given config to self.config."""
    def __init__(self, config_str):
        self.config = yaml.load(config_str)


class DummyHook(AbstractHook):  # pylint: disable=too-few-public-methods
    """Dummy dataset which save its **kwargs to self.kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)


class SecondDummyHook(AbstractHook):  # pylint: disable=too-few-public-methods
    """Second dummy dataset which does nothing."""
    pass


class EntryPointTest(CXTestCaseWithDir):
    """Entry point functions test case."""

    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_train_create_output_dir(self):
        """Test output dir creating and correct naming."""
        # test create output dir with specified net.name
        name = 'my_name'
        output_dir = train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                             output_root=self.tmpdir,
                                             default_net_name='nothing')

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_create_output_dir_no_root(self):
        """Test if output root is created if it does not exist."""
        output_root = path.join(self.tmpdir, 'output_root')
        name = 'my_name'
        output_dir = train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                             output_root=output_root,
                                             default_net_name='nothing')

        # check that output_root exists and it is the only folder in temp_dir
        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertTrue(path.exists(output_root))
        self.assertTrue(path.isdir(output_root))
        # check that output_dir exists and it is the only folder in output_root
        self.assertEqual(len(os.listdir(output_root)), 1)
        self.assertEqual(output_dir, path.join(output_root, path.basename(output_dir)))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_create_output_dir_no_netname(self):
        """Test create output dir without specified net.name (default_net_name should be used)."""
        name = 'nothing'
        output_dir = train_create_output_dir(config={'a': 'b', 'net': {}},
                                             output_root=self.tmpdir,
                                             default_net_name=name)

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_different_dirs(self):
        """Test if two calls of train_create_output_dir yields two different dirs."""
        name = 'my_name'
        output_dir_1 = train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                               output_root=self.tmpdir,
                                               default_net_name='nothing')
        output_dir_2 = train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                               output_root=self.tmpdir,
                                               default_net_name='nothing')

        self.assertNotEqual(output_dir_1, output_dir_2)
        self.assertEqual(len(os.listdir(self.tmpdir)), 2)

    def test_train_create_dataset(self):
        """Test correct config re-wrapping."""
        config = {'dataset': {'module': 'cxflow.tests.entry_point_test', 'class': 'DummyDataset', 'batch_size': 10},
                  'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

        expected_config = {'dataset': {'module': 'cxflow.tests.entry_point_test',
                                       'class': 'DummyDataset', 'batch_size': 10},
                           'stream': {'train': {'rotate': 20}}, 'output_dir': 'dummy_dir'}

        dataset = train_create_dataset(config=config, output_dir='dummy_dir')

        self.assertTrue(isinstance(dataset, DummyDataset))
        self.assertTrue(hasattr(dataset, 'config'))
        self.assertDictEqual(dataset.config, expected_config)

    def test_train_load_config(self):
        """Test correct config loading."""

        # test a config call with both dataset and net
        good_config = {'dataset': None, 'net': None}
        config_path = config_to_file(good_config, self.tmpdir)

        # test assertion when config is incomplete
        missing_net_config = {'dataset': None}
        config_path2 = config_to_file(missing_net_config, self.tmpdir, 'config2.yaml')
        self.assertRaises(AssertionError, train_load_config, config_path2, [])

        missing_dataset_config = {'dataset': None}
        config_path3 = config_to_file(missing_dataset_config, self.tmpdir, 'config3.yaml')
        self.assertRaises(AssertionError, train_load_config, config_path3, [])

        # test return value
        returned_config = train_load_config(config_path, [])
        self.assertDictEqual(returned_config, load_config(config_path, []))
        self.assertDictEqual(returned_config, good_config)

    def test_create_hooks(self):
        """Test hooks creation in train_create_hooks."""

        # test correct kwargs passing
        config = {'hooks': [{'hook_module': 'cxflow.tests.entry_point_test',
                             'hook_class':'DummyHook',
                             'additional_arg': 10}]}
        dataset = 'dataset_placeholder'
        net = 'net_placeholder'
        expected_kwargs = {'dataset': dataset, 'net': net, 'output_dir': self.tmpdir, 'additional_arg': 10}
        hooks = train_create_hooks(config=config, dataset=dataset, net=net, output_dir=self.tmpdir)
        hook = hooks[0]
        kwargs = hook.kwargs

        self.assertEqual(len(hooks), 1)
        self.assertTrue(isinstance(hook, DummyHook))
        for key in ['dataset', 'net', 'output_dir', 'additional_arg']:
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # test correct hook order
        two_hooks_config = {'hooks': [{'hook_module': 'cxflow.tests.entry_point_test',
                                       'hook_class': 'DummyHook',
                                       'additional_arg': 10},
                                      {'hook_module': 'cxflow.tests.entry_point_test',
                                       'hook_class': 'SecondDummyHook'}]}
        hooks2 = train_create_hooks(config=two_hooks_config, dataset=dataset, net=net, output_dir=self.tmpdir)

        self.assertEqual(len(hooks2), 2)
        self.assertTrue(isinstance(hooks2[0], DummyHook))
        self.assertTrue(isinstance(hooks2[1], SecondDummyHook))