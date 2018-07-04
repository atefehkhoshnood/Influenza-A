
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd

def install_pip_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def install_conda_import(channel,package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import conda.cli
        conda.cli.main('conda','install','-y','-q','-c', channel, package)
    finally:
        globals()[package] = importlib.import_module(package)

def main():
    install_conda_import('conda-forge','flaky=3.3.0')
    install_conda_import('conda-forge','joblib=0.11')
    install_conda_import('conda-forge','jupyter=1.0.0.*')
    install_conda_import('conda-forge','mdtraj=1.9.1')
    install_conda_import('rdkit','rdkit=2017.09.1')
    install_pip_import('clusterone==0.11.2')

if __name__ == "__main__":
    main()