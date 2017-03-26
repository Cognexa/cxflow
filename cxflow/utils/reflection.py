import importlib


def _create_object(module_name: str, class_name: str, **kwargs):
    """
    Create an object instance of the given class from the given module. **kwargs are passed to the object construtor.

    -----------------------------------------------------
    This mimics the following code:
    -----------------------------------------------------
    from module import class
    class(**kwargs)
    -----------------------------------------------------

    :param module_name: module name
    :param class_name: class name
    :param kwargs: kwargs to be passed to the object constructor
    :return: created object instance
    """
    module = importlib.import_module(module_name)
    _class = getattr(module, class_name)
    return _class(**kwargs)


def create_object(object_config: dict, prefix: str= '', **kwargs):
    """
    Create an object instance according to the given config.

    The config is expected to contain [prefix]module and [prefix]class.

    :param object_config: object config dict
    :param prefix: prefix of 'module' and 'class' config parameters
    :param kwargs: kwargs to be passed to the object constructor
    :return: created object instance
    """
    assert(prefix + 'module' in object_config)
    assert(prefix + 'class' in object_config)
    return _create_object(object_config[prefix + 'module'], object_config[prefix + 'class'], **kwargs)

