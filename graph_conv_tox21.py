from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv',reload=False)
train_dataset, valid_dataset, test_dataset = tox21_datasets

print(train_dataset.get_shape())
print(test_dataset.__dict__['data_dir'])
print(len(valid_dataset))
print(len(test_dataset))

model = GraphConvTensorGraph( len(tox21_tasks), batch_size=50, mode='classification')
model.fit(train_dataset, nb_epoch=10)

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')

print("Evaluating model")

train_scores = model.evaluate(train_dataset, [metric], transformers)

print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])

valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])