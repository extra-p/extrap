import inspect
import pkgutil
from .abstract_modeler import AbstractModeler


def load_modelers(path, pkg_name):
    def is_modeler(x):
        return inspect.isclass(x) \
               and issubclass(x, AbstractModeler) \
               and not inspect.isabstract(x)

    modelers = {}
    for importer, modname, is_pkg in pkgutil.walk_packages(path=path,
                                                           prefix=pkg_name + '.',
                                                           onerror=lambda x: None):
        if is_pkg:
            continue
        module = importer.find_module(modname).load_module(modname)
        for name, clazz in inspect.getmembers(module, is_modeler):
            name = clazz.NAME
            modelers[name] = clazz
    return modelers
