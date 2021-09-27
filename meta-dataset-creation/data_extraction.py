import pandas as pd
from collections import defaultdict
import numpy as np
import openml
import json

task_tag = "Verified_Supervised_Classification"

flows = openml.flows.list_flows(tag = task_tag)
flows_id = [flow["id"] for flow in flows.values()]

#### PART 1: Extraction of available run ids
step = 100 #this step is necessary as querying a lot of runs simultaneously may cause problems
runs = openml.runs.list_runs(flow = flows_id[:step])
for idx in range(0, len(flows_id),step):
    print(idx)
    temp_runs = openml.runs.list_runs(flow = flows_id[idx:idx+step])
    runs.update(temp_runs)

#Matrix to group flow id and task
matrix = defaultdict(lambda: defaultdict(list))#or 

for run in runs.values():
    matrix[run["flow_id"]][run["task_id"]].append(run['run_id']) 

df = pd.DataFrame.from_dict(data=matrix, orient='columns')

run_counts = np.zeros(df.shape)

#Keep in mind: columns are flows
for i, dataset in enumerate(list(df.index)):
    for j, flow in enumerate(list(df.columns)):
        if df[flow].loc[dataset] is not np.nan:
            run_counts[i][j] = len(df[flow].loc[dataset])

#checkpoints the created matrix in case it fails in the next steps
np.save("meta_data_checkpoint.npy", matrix)


#### PART 2: Filtering flows and datasets by specific conditions
print(np.max(run_counts))
n_min_evaluations = 5
n_min_datasets_per_flow = 1
n_min_flows_per_dataset = 1
n_flows, n_datasets = run_counts.shape


def select_elements(run_counts, n_min_evaluations, n_min_datasets_per_flow, n_min_flows_per_dataset):

    n_datasets, n_flows = run_counts.shape
    valid_evaluations_flag = run_counts>n_min_evaluations
    valid_flows_flag = np.sum(valid_evaluations_flag, axis=0)>n_min_datasets_per_flow
    valid_dataset_flag = np.sum(valid_evaluations_flag, axis=1)>n_min_flows_per_dataset
    dataset_flag = np.tile(np.array(valid_dataset_flag), (n_flows,1)).T
    flow_flag = np.tile(np.array(valid_flows_flag), (n_datasets,1))

    selected_elements = np.multiply(valid_evaluations_flag, dataset_flag, flow_flag)
    selected_datasets, selected_flows = np.where(selected_elements>0)


    n_valid_flows =  np.unique(selected_flows).shape[0]
    n_valid_datasets = np.unique(selected_datasets).shape[0]

    return selected_flows, selected_datasets, n_valid_flows, n_valid_datasets

selected_flows, selected_datasets, n_flows, n_datasets= select_elements(run_counts, n_min_evaluations, n_min_datasets_per_flow, n_min_flows_per_dataset)

selected_flows_ids = df.columns[selected_flows] 
print("Selected flows ids:", selected_flows_ids)

selected_datasets_ids = df.index[selected_datasets] 
print("Selected datasets ids:", selected_datasets_ids)


### PART 3: Create the raw version of the meta-dataset


def process_run_settings (settings):
    parameter_dict ={}
    for setting in settings:
        value = setting["oml:value"]
        parameter_dict[setting["oml:name"]] = value

    return parameter_dict


meta_dataset = defaultdict(lambda: defaultdict(set))

#Defines the starting index of the selected datasets and flows. 
#In case the extraction takes too long, it is recommended to use several starting idx
starting_idx = 0

for idx, (dataset, flow) in enumerate(zip(selected_datasets_ids[starting_idx:], selected_flows_ids[starting_idx:])):
    print("Processing dataset:", dataset, " and flow: ", flow, " position index:",idx)
    meta_dataset[dataset][flow] = []
    
    
    run_list = list(matrix[flow][dataset])
    
    for run_list_idx in range(0, max(2000,len(run_list)), 1000):

        
        current_runs = openml.runs.get_runs(list(run_list)[run_list_idx:run_list_idx+1000])

        for i, current_run in enumerate(current_runs):

            if (i%100==0):
                print("Processing :", i, " out of ", len(matrix[flow][dataset]), " from running list idx:", run_list_idx )


            temp_dict = {}
            temp_dict["run_id"] = run_list[i]
            temp_dict["task_id"] = current_run.task_id
            temp_dict["flow_name"] = current_run.flow_name
            temp_dict["accuracy"] = current_run.evaluations["predictive_accuracy"]
            temp_dict["parameter_settings"] =  process_run_settings (current_run.parameter_settings)
            meta_dataset[dataset][flow].append(temp_dict)


with open('meta_data_openml.json', 'w') as outfile:
    json.dump(meta_dataset, outfile)