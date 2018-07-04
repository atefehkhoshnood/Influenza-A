
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

conda install -y -q -c omnia pdbfixer=1.4
conda install -y -q -c deepchem mdtraj=1.9.1
conda install -y -q -c rdkit rdkit=2017.09.1
conda install -y -q -c conda-forge joblib=0.11 \
    six=1.11.0 \
    scikit-learn=0.19.1 \
    networkx=2.1 \
    pillow=5.0.0 \
    pandas=0.22.0 \
    nose=1.3.7 \
    nose-timer=0.7.0 \
    flaky=3.3.0 \
    zlib=1.2.11 \
    requests=2.18.4 \
    xgboost=0.6a2 \
    simdna=0.4.2 \
    jupyter=1.0.0 \
    pbr=3.1.1 \
    setuptools=39.0.1 \
    biopython=1.71 \
    numpy=1.14

pip install clusterone==0.11.2

import sklearn
import rdkit 
import time
import os
import logging
import traceback
import json
import glob
import tensorflow as tf
import deepchem as dc
import numpy as np
import pandas as pd
import h5py

from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

# What is your ClusterOne username? This should be something like "johndoe", not your email address!
CLUSTERONE_USERNAME = "atefeh"

# Where should your local log files be stored? This should be something like "~/Documents/self-driving-demo/logs/"
LOCAL_LOG_LOCATION = "/Users/MAJ/Desktop/DeepChem"

# Where is the dataset located? This should be something like "~/Documents/data/" if the dataset is in "~/Documents/data/comma"
LOCAL_DATASET_LOCATION = "/Users/MAJ/Desktop/"

# Name of the data folder. In the example above, "comma"
LOCAL_DATASET_NAME = "DeepChem"

#clusterone
from clusterone import get_data_path, get_logs_path

def main():

    try:
        job_name = os.environ['JOB_NAME']
        task_index = os.environ['TASK_INDEX']
        ps_hosts = os.environ['PS_HOSTS']
        worker_hosts = os.environ['WORKER_HOSTS']
    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None

    if job_name == None: #if running locally
        if LOCAL_LOG_LOCATION == "...":
            raise ValueError("LOCAL_LOG_LOCATION needs to be defined")
        if LOCAL_DATASET_LOCATION == "...":
            raise ValueError("LOCAL_DATASET_LOCATION needs to be defined")
        if LOCAL_DATASET_NAME == "...":
            raise ValueError("LOCAL_DATASET_NAME needs to be defined")

#Path to your data locally. This will enable to run the model both locally and on
# ClusterOne without changes
    PATH_TO_LOCAL_LOGS = os.path.expanduser(LOCAL_LOG_LOCATION)
    ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser(LOCAL_DATASET_LOCATION)
#end of clusterone snippet 1


if __name__ == "__main__":
    main()