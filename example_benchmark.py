
import numpy as np
import json
from benchmark_plot import BenchmarkPlotter
from methods.pygpgo import RandomForest



if __name__ == "__main__":

    data_path = "hpob-data/"
    results_path = "results/"
    output_path = "plots/"
    name = "RF_benchmark"
    new_method_name = "RF.json"
    experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP"]
    n_trials = 5

    method = RandomForest(acq_name="ProbabilityImprovement")

    benchmark_plotter  = BenchmarkPlotter(experiments = experiments, 
                                            name = name,
                                            n_trials = n_trials,
                                            results_path = results_path, 
                                            output_path = output_path, 
                                            data_path = data_path)

    benchmark_plotter.generate_results(method, n_trials, new_method_name)
    benchmark_plotter.plot()
    benchmark_plotter.draw_cd_diagram(bo_iter=5, name="Rank@5")
