import os

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def get_project_root():
    module_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(module_path, '..'))

def get_cached_data(filename):
    os.makedirs()
    root = get_project_root()
    os.path.join(root, '.cache', 'data', )