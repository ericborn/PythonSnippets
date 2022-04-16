# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:43:40 2022

@author: Eric
# 
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import json
import numpy as np

# log reg to json
def logistic_regression_to_json(lrmodel, file=None):
    if file is not None:
        serialize = lambda x: json.dump(x, file)
    else:
        serialize = json.dumps
    data = {}
    data['init_params'] = lrmodel.get_params()
    data['model_params'] = mp = {}
    for p in ('coef_', 'intercept_','classes_', 'n_iter_'):
        mp[p] = getattr(lrmodel, p).tolist()
    return serialize(data)

# log reg from json
def logistic_regression_from_json(jstring):
    file = open(jstring)
    data = json.load(file)
    file.close()
    model = LogisticRegression(**data['init_params'])
    for name, p in data['model_params'].items():
        setattr(model, name, np.array(p))
    return model

# scaler to json
def scaler_to_json(lrscaler, file=None):
    if file is not None:
        serialize = lambda x: json.dump(x, file)
    else:
        serialize = json.dumps
    data = {}
    data['init_params'] = lrscaler.get_params()
    data['scaler_params'] = sp = {}
    for p in ('scale_', 'mean_','var_'):
        sp[p] = getattr(lrscaler, p).tolist()
    return serialize(data)

# scale from json
def scaler_from_json(jstring):
    file = open(jstring)
    data = json.load(file)
    file.close()
    scaler = StandardScaler()
    scaler.set_params(**data['init_params'])
    for name, p in data['scaler_params'].items():
        setattr(scaler, name, np.array(p))
    return scaler

# path to pickle files
model = joblib.load(r'/home/eric/Documents/autodj/src/autodj/annotation/downbeat/model.pkl') 
scaler = joblib.load(r'/home/eric/Documents/autodj/src/autodj/annotation/downbeat/scaler.pkl')

# json files
model_json = "model.json"	
scaler_json = "scaler.json"

# save model to json
model_file = open(model_json, 'w')
logistic_regression_to_json(model, model_file)
model_file.close()

# save scaler to json
scaler_file = open(scaler_json, 'w')
scaler_to_json(scaler, scaler_file)
scaler_file.close()

# generate new model from json
new_model = logistic_regression_from_json('model.json')

# generate new scaler from json
new_scaler = scaler_from_json('scaler.json')