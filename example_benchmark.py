
import numpy as np
import json
from benchmark_plot import BenchmarkPlotter
from hpob_handler import HPOBHandler
from methods.pygpgo import RandomForest

data_path = "hpob-data/"
generate_results = True
n_trials = 5

if generate_results:
    hpob_hdlr = HPOBHandler(root_dir=data_path, mode="v3-test")
    method = RandomForest(acq_name="ProbabilityImprovement")

    rf_results = {}

    for search_space_id in hpob_hdlr.get_search_spaces():

        if search_space_id not in rf_results.keys():
            rf_results[search_space_id] = {} 

        for dataset_id in hpob_hdlr.get_datasets(search_space_id):

            if dataset_id not in rf_results[search_space_id].keys():
                rf_results[search_space_id][dataset_id] = {} 

            for seed in hpob_hdlr.get_seeds():

                rf_results[search_space_id][dataset_id][seed]  = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                        dataset_id = dataset_id,
                                                        seed = seed,
                                                        n_trials = n_trials )
                                                    

    with open("results/RF.json", "w") as f:
        json.dump(rf_results, f)

results_path = "results/"
output_path = "plots/"
experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP", "RF"]

benchmark_plotter  = BenchmarkPlotter(experiments=experiments, 
                                        max_bo_iters = n_trials+1,
                                        results_path=results_path, 
                                        output_path=output_path, 
                                        data_path = data_path)

benchmark_plotter.plot()
benchmark_plotter.draw_cd_diagram(bo_iter=5, name="Rank@5")
