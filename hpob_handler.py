#Before running this notebook download the data from https://rewind.tf.uni-freiburg.de/index.php/s/B6gY9cpZ65fBfGJ
#For easing the consumption of the data
import numpy as np
import pandas as pd
import json
import os


class HPOBHandler:

    def __init__(self, root_dir = "", mode = "test"):
        print("Loading boss handler")
        self.mode = mode

        if self.mode == "test":
            self.load_data(root_dir, only_test=True)
        elif self.mode == "train-augmented":
            self.load_data(root_dir, only_test=False, augmented_train=True)
        else:
            self.load_data(root_dir, only_test=False)

    def load_data(self, rootdir="", only_test = True, augmented_train = False):

        print("Loading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            self.meta_test_data = json.load(f)
        
        with open(bo_initializations_path, "rb") as f:
            self.bo_initializations = json.load(f)

        if not only_test:

            if augmented_train:
                with open(meta_train_augmented_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                self.meta_validation_data = json.load(f)

    def normalize(self, y):

        return (y-np.min(y))/(np.max(y)-np.min(y))

    def evaluate (self, bo_method = None, search_space_id = None, dataset_id = None, trial = None, n_iterations = 10):

        assert bo_method!=None, "Provide a valid method object for evaluation."
        assert hasattr(bo_method, "observe_and_suggest"), "The provided  object does not have a method called ´observe_and_suggest´"
        assert search_space_id!= None, "Provide a valid search space id. See documentatio for valid obptions."
        assert dataset_id!= None, "Provide a valid dataset_id. See documentation for valid options."
        assert trial!=None, "Provide a valid initialization. Valid options are: test0, test1, test2, test3, test4."

        n_initial_evaluations = 5
        X = np.array(self.meta_test_data[search_space_id][dataset_id]["X"])
        y = np.array(self.meta_test_data[search_space_id][dataset_id]["y"])
        y = self.normalize(y)
        data_size = len(X)
        
        pending_evaluations = list(range(data_size))
        current_evaluations = []        

        init_ids = self.bo_initializations[search_space_id][dataset_id][trial]
        
        for i in range(n_initial_evaluations):
            idx = init_ids[i]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)

        max_performance_history = []
        for i in range(n_iterations):

            idx = bo_method.observe_and_suggest(X[current_evaluations], y[current_evaluations], X[pending_evaluations])
            idx = pending_evaluations[idx]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)
            max_performance_history.append(np.max(y[current_evaluations]))
        
        return max_performance_history

    def get_search_spaces(self):
        return list(self.meta_test_data.keys())

    def get_datasets(self, search_space):
        return list(self.meta_test_data[search_space].keys())
