
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
#import deepchem as dc
import numpy as np
import pandas as pd

I = [(1,2),(3,4)]
print(I)
Print('Hello!')
#from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

#data_path = get_data_path(
#           dataset_name = "/Users/MAJ/Desktop/DeepChem/Influenza-A-virus-Active-NotActive.csv",
#           local_root = te_path,
#           local_repo = te_filename,
#           path = ''
#           )

#graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
#loader = dc.data.data_loader.CSVLoader( tasks=['ACTIVITY_CLASS'], smiles_field="CANONICAL_SMILES", id_field="CMPD_CHEMBLID", featurizer=graph_featurizer)
#dataset = loader.featurize( 'Influenza-A-virus-Active-NotActive.csv' )
 
#splitter = dc.splits.splitters.RandomSplitter()
#train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

#prob_tasks = ['ACTIVITY_CLASS']
#model = GraphConvTensorGraph(
#    len(prob_tasks), batch_size=50, mode='classification')
# Set nb_epoch=10 for better results.
#scores = []
#for i in range(1,5,1):
#    model.fit(train_dataset, nb_epoch=i)
#    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
#    train_scores = model.evaluate(train_dataset, [metric])
#    valid_scores = model.evaluate(valid_dataset, [metric])
#    scores.append((train_scores,valid_scores))
#print(scores)

#prediction_test = model.predict(test_dataset)

#import pandas as pd
#df_2 = pd.read_csv('Influenza-A-virus-Active-NotActive.csv', header=0)
#smiles = []
#for i in range(len(df_2.index)):
#    smiles.append(len(df_2.loc[i,'CANONICAL_SMILES']))
#model.predict_on_smiles(smiles)

#test_dataset.get_shape()

#for i in range(len(test_dataset.y)):
 #   if test_dataset.y[i] == 1:
  #      print(test_dataset.ids[i], test_dataset.y[i], prediction_test[i])