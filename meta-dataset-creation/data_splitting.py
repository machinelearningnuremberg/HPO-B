import json
import numpy as np
import pandas as pd


with open("meta_dataset.json") as f:
    meta_dataset = json.load(f)

search_spaces = list(meta_dataset.keys())

for flow in meta_dataset.keys():
    for dataset in meta_dataset[flow].keys():
        
        df_X = pd.DataFrame(meta_dataset[flow][dataset]["X"])
        df_y = pd.DataFrame(meta_dataset[flow][dataset]["y"])
        
        duplicated = df_X.duplicated()
        X = df_X[~duplicated]
        y = df_y[~duplicated]
        meta_dataset[flow][dataset] = {"X": X.to_numpy().tolist(), "y": y.to_numpy().tolist()}


n_min_datasets = 10
train_pct = 0.8
val_pct = 0.9

meta_dataset_train_augmented = {}
meta_dataset_train = {}
meta_dataset_validation = {}
meta_dataset_test = {}

for flow in meta_dataset.keys():

    datasets = list(meta_dataset[flow].keys())
    n_datasets = len(datasets)
    
    datasets_shape = []
    for dataset in datasets:
        datasets_shape.append(len(meta_dataset[flow][dataset]["X"]))
    
    sorted_datasets = np.array(datasets)[np.argsort(datasets_shape)]
    sorted_shape = np.array(datasets_shape)[np.argsort(datasets_shape)]
    
   
    print("For flow ", flow, " number of datasets:", n_datasets)
    i0 = int(np.ceil(train_pct*n_datasets))
    i1 = int(np.ceil(val_pct*n_datasets))

    
    meta_dataset_train_augmented[flow] = {}
    meta_dataset_train[flow] = {}
    
    if n_datasets>np.ceil(train_pct*n_datasets):
        meta_dataset_validation[flow] = {}
    
    if n_datasets>n_min_datasets and 99<sorted_shape[i1]:
        meta_dataset_test[flow] = {}
        

    
    for i, dataset in enumerate(sorted_datasets.tolist()):
        
        df = pd.DataFrame(meta_dataset[flow][dataset]["X"])
        meta_dataset[flow][dataset]["X"] = df.to_numpy().tolist()
        
        if n_min_datasets>n_datasets:
            meta_dataset_train_augmented[flow][dataset] = meta_dataset[flow][dataset]
            
        else:

            if i<i0:

                if n_datasets>n_min_datasets and 99<sorted_shape[i1]:
                    meta_dataset_train[flow][dataset] = meta_dataset[flow][dataset]
                else:

                    meta_dataset_train_augmented[flow][dataset] = meta_dataset[flow][dataset]


            elif i<i1:

                if n_datasets>n_min_datasets and 99<sorted_shape[i1]:
                    meta_dataset_validation[flow][dataset] = meta_dataset[flow][dataset]
                else:
                    meta_dataset_train_augmented[flow][dataset] = meta_dataset[flow][dataset]


            else:

                if n_datasets>n_min_datasets and 99<sorted_shape[i1]:
                    meta_dataset_test[flow][dataset] = meta_dataset[flow][dataset]   
                else:
                    meta_dataset_train_augmented[flow][dataset] = meta_dataset[flow][dataset]


with open('meta-train-dataset.json', 'w') as outfile:
    json.dump(meta_dataset_train, outfile)

with open('meta-test-dataset.json', 'w') as outfile:
    json.dump(meta_dataset_test, outfile)

with open('meta-validation-dataset.json', 'w') as outfile:
    json.dump(meta_dataset_validation, outfile)

with open('meta-train-dataset-augmented.json', 'w') as outfile:
    json.dump(meta_dataset_train_augmented, outfile)
