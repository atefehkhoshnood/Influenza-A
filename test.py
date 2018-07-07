
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import time
import os
#import tensorflow as tf
import numpy as np
import pandas as pd

def install_pip_import(package,version):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        package_version=package+'=='+version
        print(package_version)
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def install_pip(package,version):
    import pip
    package_version=package+'=='+version
    print(package_version)
    pip.main(['install', package])
    

def install_conda_import(channel,package,version):
    import importlib
    import conda
    print('conda version is:'+conda.__version__)
    try:
        importlib.import_module(package)
    except ImportError:
        import conda.cli
        package_version=package+'='+version
        print(package_version)
        conda.cli.main('conda','install','-y','-v -v','-q','-c', channel, package_version)
    finally:
        globals()[package] = importlib.import_module(package)

def install_conda(channel,package,version):
    import conda.cli
    import conda
    print('conda version is:'+conda.__version__)
    package_version=package+'='+version
    print(package_version)
    conda.cli.main('conda','install','-y','--debug','-q','-c', channel, package_version)
    print('installed:'+package_version)
 
def main():
#    install_conda_import('conda-forge','flaky','3.3.0')
#    install_conda_import('conda-forge','joblib','0.11')
#    install_conda_import('conda-forge','jupyter','1.0.0.*')
#    install_conda_import('deepchem','mdtraj','1.9.1')
#    install_conda_import('conda-forge','networkx','1.11')
#    install_conda_import('conda-forge','nose','1.3.7')
#    install_conda_import('conda-forge','nose-timer','0.7.0')
#    install_conda_import('conda-forge','pandas','0.22.0')
#    install_conda_import('omnia','pdbfixer','1.4')
#    install_conda_import('conda-forge','pillow','4.3.0')
#    install_conda_import('conda-forge','python','>=2.7,<2.8.0a0')
#    install_conda_import('conda-forge','scikit-learn','0.18.1')
#    install_conda_import('conda-forge','simdna','0.4.2')
#    install_conda_import('conda-forge','requests','2.18.4')
#    install_conda_import('conda-forge','xgboost','0.6a2')
#    install_conda_import('conda-forge','zlib','1.2.11')
#    install_conda_import('conda-forge','h5py','2.7.1')
#    install_conda_import('conda-forge','numpy','1.13.3')
#    install_conda_import('rdkit','rdkit','2017.09.1')
#    install_conda_import('','deepchem','2.0.0')

    install_pip_import('conda','4.5.4')
    install_pip('ruamel.yaml','0.15.42')
    install_pip('pyyaml','3.12')

    install_pip_import('flaky','3.3.0')
    install_pip_import('joblib','0.11')
    install_pip_import('jupyter','1.0.0.*')
    install_pip_import('numpy','1.13.3')
    install_pip_import('cython','>=0.19')
    install_pip_import('mdtraj','1.9.1')
    install_pip_import('networkx','1.11')
    install_pip_import('nose','1.3.7')
    install_pip('nose-timer','0.7.0')
    install_pip_import('pandas','0.22.0')
#    install_pip_import('pdbfixer','1.4')
    install_conda('omnia','pdbfixer','1.4')
    install_pip('pillow','4.3.0')
    install_pip('python','>=2.7,<2.8.0a0')
    install_pip('scikit-learn','0.18.1')
    install_pip_import('simdna','0.4.2')
    install_pip_import('requests','2.18.4')
    install_pip_import('xgboost','0.6a2')
    install_pip_import('zlib','1.2.11')
    install_pip_import('h5py','2.7.1')
#    install_pip_import('rdkit','2017.09.1')
    install_conda_import('rdkit','rdkit','2017.09.1')
    install_pip_import('deepchem','2.0.0')

    install_pip_import('clusterone','0.11.2')

if __name__ == "__main__":
    main()