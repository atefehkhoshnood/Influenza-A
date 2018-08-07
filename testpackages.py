
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def main():
    import tensorflow as tf
    print('Tensorflow imported')
    print('Tensorflow version', tf.__version__)

    import deepchem as dc
    print('deepchem imported')
    print('deepchem version', dc.__version__)

if __name__ == "__main__":
    main()