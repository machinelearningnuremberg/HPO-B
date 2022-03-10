
import numpy as np
import json
from benchmark_plot import BenchmarkPlotter
from hpob_handler import HPOBHandler
from methods.pygpgo import RandomForest


def generate_results (method, results_path, data_path, n_trials):

    hpob_hdlr = HPOBHandler(root_dir=data_path, mode="v3-test")
    results = {}

    for search_space_id in hpob_hdlr.get_search_spaces():

        if search_space_id not in results.keys():
            results[search_space_id] = {} 
        
        for dataset_id in hpob_hdlr.get_datasets(search_space_id):

            if dataset_id not in results[search_space_id].keys():
                results[search_space_id][dataset_id] = {} 

            for seed in hpob_hdlr.get_seeds():

                results[search_space_id][dataset_id][seed]  = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                        dataset_id = dataset_id,
                                                        seed = seed,
                                                        n_trials = n_trials )
                                                    

    with open(results_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    data_path = "hpob-data/"
    results_path = "results/"
    output_path = "plots/"
    name = "RF_benchmark"
    results_name = "RF.json"
    experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP", "RF"]
    n_trials = 5

    method = RandomForest(acq_name="ProbabilityImprovement")
    generate_results(method, results_path+results_name, data_path, n_trials)

    benchmark_plotter  = BenchmarkPlotter(experiments=experiments, 
                                            name = name,
                                            max_bo_iters = n_trials+1,
                                            results_path=results_path, 
                                            output_path=output_path, 
                                            data_path = data_path)

    benchmark_plotter.plot()
    benchmark_plotter.draw_cd_diagram(bo_iter=5, name="Rank@5")
