"""
Test module for reflection utils (cxflow.utils.reflection).
"""
import sys
import os
import pytest

from cxflow.utils.reflection import create_object, find_class_module, get_class_module, parse_fully_qualified_name


class SimpleClass:  # pylint: disable=missing-docstring
    pass


class DuplicateClass:  # pylint: disable=missing-docstring
    pass


class ImportedClass:  # pylint: disable=missing-docstring
    # this class is imported in cxflow.tests.utils.dummy_module
    pass


class ClassWithArg:  # pylint: disable=missing-docstring
    def __init__(self, x):
        self.ex = x


class ClassWithArgs:  # pylint: disable=missing-docstring
    def __init__(self, x, *args):
        self.ex = x
        self.args = args


class ClassWithKwargs:  # pylint: disable=missing-docstring
    def __init__(self, x, **kwargs):
        self.ex = x
        self.kwargs = kwargs


class ClassWithArgsAndKwargs:  # pylint: disable=missing-docstring
    def __init__(self, x, *args, **kwargs):
        self.ex = x
        self.args = args
        self.kwargs = kwargs


class ClassWithArgsAndKwargsOnly:  # pylint: disable=missing-docstring
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_parse_fully_qualified_name():
    """Test correct parsing of fully qualified names."""

    # test simple name
    module1, class1 = parse_fully_qualified_name('MyClass')
    assert module1 is None
    assert class1 == 'MyClass'

    # test simple path
    module1, class1 = parse_fully_qualified_name('MyModule.MyClass')
    assert module1 == 'MyModule'
    assert class1 == 'MyClass'

    # test complex path
    module1, class1 = parse_fully_qualified_name('MyModule.MySubmodule.MyClass')
    assert module1 == 'MyModule.MySubmodule'
    assert class1 == 'MyClass'


def test_create_object():
    """Test base create object function."""
    module_name = 'cxflow.tests.utils.reflection_test'

    # test type
    obj0 = create_object(module_name=module_name, class_name='SimpleClass')
    assert type(obj0) == SimpleClass

    obj = create_object(module_name=module_name, class_name='ClassWithArg', args=(12,))
    assert type(obj) == ClassWithArg
    assert obj.ex == 12

    # test args and kwargs
    obj2 = create_object(module_name=module_name, class_name='ClassWithArgs', args=(12, 1, 2, 3))
    assert obj2.ex == 12
    assert obj2.args == (1, 2, 3)

    obj3 = create_object(module_name=module_name, class_name='ClassWithKwargs', kwargs={'x': 12, 'y': 1, 'z': 2})
    assert obj3.ex == 12
    assert obj3.kwargs == {'y': 1, 'z': 2}

    obj4 = create_object(module_name=module_name, class_name='ClassWithKwargs', args=(12,), kwargs={'y': 1, 'z': 2})
    assert obj4.ex == 12
    assert obj4.kwargs == {'y': 1, 'z': 2}

    obj5 = create_object(module_name=module_name, class_name='ClassWithArgsAndKwargs',
                         args=(12, 1, 2, 3), kwargs={'y': 1, 'z': 2})
    assert obj5.ex == 12
    assert obj5.args == (1, 2, 3)
    assert obj5.kwargs == {'y': 1, 'z': 2}

    obj6 = create_object(module_name=module_name, class_name='ClassWithArgsAndKwargsOnly',
                         args=(1, 2, 3, ), kwargs={'y': 1, 'z': 2})
    assert obj6.args == (1, 2, 3)
    assert obj6.kwargs == {'y': 1, 'z': 2}


INVALID_PARAMETERS = [
    ({}, TypeError),
    ({'module_name': 'cxflow.tests.utils.reflection_test'}, TypeError),
    ({'class_name': 'SimpleClass'}, TypeError),  # insufficient arguments
    ({'module_name': 123, 'class_name': 'SimpleClass'}, AssertionError),
    ({'module_name': 'cxflow.tests.utils.reflection_test', 'class_name': 123}, AssertionError),  # wrong argument types
    ({'module_name': 'cxflow.tests.utils.reflection_test_xxx', 'class_name': 'SimpleClass'}, ImportError),
    ({'module_name': 'cxflow.tests.utils.reflection_test', 'class_name': 'WrongName'},
     AttributeError)  # wrong module/class names
]


@pytest.mark.parametrize('params, error', INVALID_PARAMETERS)
def test_create_object_fails(params, error):
    """Test create object behavior with erroneous aruments."""

    with pytest.raises(error):
        create_object(**params)


FIND_MODULES = [(['SimpleClass'], ['cxflow.tests.utils.reflection_test']),  # correct setup
                (['DoesNotExists'], []),  # non-existent class
                (['DuplicateClass'],
                 ['cxflow.tests.utils.reflection_test', 'cxflow.tests.utils.dummy_module'])]  # multiple matches


@pytest.mark.parametrize('params, expected_modules', FIND_MODULES)
def test_find_class_module(params, expected_modules):
    """Test finding class module."""

    matched_modules, erroneous_modules = find_class_module('cxflow.tests.utils', *params)
    assert sorted(matched_modules) == sorted(expected_modules)
    assert erroneous_modules == []


def test_find_class_module_errors(tmpdir):
    """Test erroneous modules handling in find_class_module function."""

    # create dummy module hierarchy
    module_name = 'my_dummy_module'
    valid_submodule_name = 'my_valid_submodule'
    invalid_submodule_name = 'my_invalid_submodule'
    module_path = os.path.join(tmpdir, module_name)

    os.mkdir(module_path)
    with open(os.path.join(module_path, '__init__.py'), 'w') as file:
        file.write('\n')
    with open(os.path.join(module_path, invalid_submodule_name+'.py'), 'w') as file:
        file.write('import ANonExistentModule\n')
    with open(os.path.join(module_path, valid_submodule_name+'.py'), 'w') as file:
        file.write('class MyClass:\n    pass\n')

    sys.path.append(str(tmpdir))

    matched_modules, erroneous_modules = find_class_module(module_name, 'MyClass')

    # test if the correct module is returned despite the erroneous module
    assert matched_modules == [module_name+'.'+valid_submodule_name]

    # test if the erroneous module is returned correctly
    assert len(erroneous_modules) == 1
    assert erroneous_modules[0][0] == module_name+'.'+invalid_submodule_name
    assert isinstance(erroneous_modules[0][1], ImportError)


def test_get_class_module():
    """Test if get_class_module method wraps the `utils.reflection.find_class_module` method correctly."""

    # test if the module is returned directly
    module = get_class_module('cxflow.hooks', 'LogProfile')
    expected_module = 'cxflow.hooks.log_profile'
    assert module == expected_module

    # test if None is returned when the class is not found
    module2 = get_class_module('cxflow.hooks', 'IDoNotExist')
    expected_module2 = None
    assert module2 == expected_module2

    # test if exception is raised when multiple modules are matched
    with pytest.raises(ValueError):
        get_class_module('cxflow.tests.utils', 'DuplicateClass')

    # test if any of valid sub-modules is returned if their attribute points to the same class
    # e.g. one sub-module imports a class from another sub-module
    module3 = get_class_module('cxflow.tests.utils', 'ImportedClass')
    possible_submodules = {'cxflow.tests.utils.reflection_test', 'cxflow.tests.utils.dummy_module'}
    assert module3 in possible_submodules
