#code for training surrogates (unpolished)

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns; sns.set_theme()
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from xgboost import cv

def train_and_save_xgb(X, y, surrogates_path="saved-surrogates/", model_name="surrogate-X.", num_round = 10, param={'max_depth':6, 'eta':0.1, 'objective':'reg:squarederror' } , logs_file="logs/train-info.txt"):

    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtrain)    
    error = mse(preds, y)
    bst.save_model(surrogates_path+model_name+".json")

    with open(logs_file, "a") as file_object:
        file_object.write("MSE="+str(round(error, 5))+" for "+model_name+ " with eta="+ str(param["eta"]) + " and rounds=" +str(num_round)+ "\n")

def tune_hyperparemeters(data_matrix, objective):
   
    grid_search_results = []
    for eta in [0.001, 0.01, 0.1, 0.5]:
        for rounds in [ 20, 50, 100, 500]:
            params ={'max_depth':6, 'eta':eta, 'objective':objective }
            xgb_cv = cv(dtrain=data_matrix, params=params, nfold=5, num_boost_round=rounds, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
            grid_search_results.append((eta, rounds, np.array(xgb_cv["test-rmse-mean"])[-1]))

    amin=np.array(grid_search_results)[:,-1].argmin()
    best_eta, best_rounds = np.array(grid_search_results)[amin,:2] 

    return best_eta, best_rounds

num_round = 10
data_path = "../transfer-HPO/data/hpob-data/"
surrogates_path = "saved-surrogates/"
objective = 'reg:squarederror'
#objective = 'reg:logistic'
normalize = False
logs_file = "logs/train-info.txt"

with open(data_path+"/meta-train-dataset-augmented.json", "r") as f:
    hpo_data_train = json.load(f) 

with open(data_path+"/meta-test-dataset.json", "r") as f:
    hpo_data_test = json.load(f) 

with open(data_path+"/meta-validation-dataset.json", "r") as f:
    hpo_data_validation = json.load(f) 

for hpo_data in [hpo_data_train, hpo_data_test, hpo_data_validation]:
    for search_space in hpo_data.keys():
        for task in hpo_data[search_space].keys():
            

            print("Training surrogate for search space:", search_space, " and task ", task)
            X = np.array(hpo_data[search_space][task]["X"])
            y = np.array(hpo_data[search_space][task]["y"])

            if normalize:
                y = MinMaxScaler().fit_transform(y)
            data_matrix = xgb.DMatrix(X, label=y)
            eta, rounds = tune_hyperparemeters(data_matrix, objective)
            param={'max_depth':6, 'eta':eta, 'objective': objective} 
            train_and_save_xgb(X,y, surrogates_path=surrogates_path, model_name="surrogate-"+search_space+"-"+task, num_round=int(rounds), param=param, logs_file=logs_file)



