
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import pandas
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

graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader( tasks=['ACTIVITY_CLASS'], smiles_field="CANONICAL_SMILES", id_field="CMPD_CHEMBLID", featurizer=graph_featurizer)
dataset = loader.featurize( 'Influenza-A-virus-Active-NotActive.csv' )
 
splitter = dc.splits.splitters.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

prob_tasks = ['ACTIVITY_CLASS']
model = GraphConvTensorGraph(
    len(prob_tasks), batch_size=50, mode='classification')
# Set nb_epoch=10 for better results.
scores = []
for i in range(1,5,1):
    model.fit(train_dataset, nb_epoch=i)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
    train_scores = model.evaluate(train_dataset, [metric])
    valid_scores = model.evaluate(valid_dataset, [metric])
    scores.append((train_scores,valid_scores))
print(scores)

prediction_test = model.predict(test_dataset)

for i in range(len(test_dataset.y)):
	if test_dataset.y[i] == 1:
		print(test_dataset.ids[i], test_dataset.y[i], prediction_test[i])