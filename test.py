
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import pip
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def main():
    install_and_import('rdkit')
    install_and_import('clusterone')

if __name__ == "__main__":
    main()