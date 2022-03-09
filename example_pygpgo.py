import matplotlib.pyplot as plt
import numpy as np
from hpob_handler import HPOBHandler
from methods.pygpgo import RandomForest

valid_acquisitions = ["Entropy", "ExpectedImprovement", "IntegratedExpectedImprovement", "ProbabilityImprovement", "IntegratedProbabilityImprovement", "UCB", "IntegratedUCB"]
seeds = ["test0", "test1", "test2", "test3", "test4"]
acc_list = []
n_trials = 20

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")
search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[0]

for acq_name in valid_acquisitions:
    acc_per_method = []
    for seed in seeds:
        print("Using ", acq_name, " as acquisition function...")

        #define the HPO method
        method = RandomForest(acq_name=acq_name)

        #evaluate the HPO method
        acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                seed = seed,
                                                n_trials = n_trials )
        acc_per_method.append(acc)

    plt.plot(np.array(acc_per_method).mean(axis=0))
plt.legend(valid_acquisitions)
plt.show()
