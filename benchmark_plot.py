import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from cd_diagram import draw_cd_diagram as draw
from matplotlib.ticker import MaxNLocator
from hpob_handler import HPOBHandler

class BenchmarkPlotter:

    def __init__(self, experiments=None, seeds = None, draw_std=True, draw_per_space=True, n_trials = 100, 
                    name="benchmark_plot", output_path="plots/", 
                    results_path = "results/", data_path = "data/"):
        
        super(BenchmarkPlotter, self).__init__()

        assert experiments is not None, "Provide the name of the experiments to plot"
        assert n_trials<101,"The maximum value for max_bo_iters is 101" 

        self.experiments = experiments
        self.seeds = seeds if seeds is not None else ["test0", "test1", "test2", "test3", "test4"]
        self.draw_std = draw_std
        self.draw_per_space = draw_per_space
        self.path = output_path
        self.name = name
        self.results_path = results_path
        self.data_path = data_path
        self.n_trials = n_trials + 1

        self.load_results()

        with open(data_path+"meta-test-tasks-per-space.json", "r") as f:
            self.task_list_per_space = json.load(f) 

        self.search_spaces = list(self.task_list_per_space.keys())

    def plot(self):

        self.generate_rank_and_regret()
        self.generate_plots_per_search_space( name= self.name+"_per_space")
        self.generate_aggregated_plots(name= self.name+"_aggregated")

    def make_rank_and_regret_plot(self, rank_list, regret_list, axis_rank, axis_regret, title=""):

        rank = np.array(rank_list)
        regret = np.array(regret_list)
        sample_size, n_experiments, n_bo_iters = rank.shape

        rank_mean = np.nanmean(rank,axis=0)
        regret_mean = np.nanmean(regret,axis=0)
        rank_std = np.nanstd(rank,axis=0)
        regret_std = np.nanstd(regret,axis=0)
        ci_factor = 1.96/np.sqrt(sample_size)

        self.plots_on_axis(axis_rank, rank_mean, rank_std, ci_factor, title, "Average Rank", self.draw_std)
        self.plots_on_axis(axis_regret, regret_mean, regret_std, ci_factor, title, "Average Regret",  self.draw_std, scale="log")

    def load_results(self):

        self.results = {}

        for experiment in self.experiments:
            with open(self.results_path+experiment+".json") as f:
                temp_data = json.load(f)
            self.results[experiment] =  temp_data
        
        return self.results

    def plots_on_axis(self, axis, mean, std, ci_factor, title="", y_label="Average Rank", draw_std=False, scale = "linear"):


        for k in range(mean.shape[0]):
            x = mean[k,:]
            axis.plot(x, linewidth=5)

            if draw_std:
                x_std = std[k,:]*ci_factor*0.5
                ci1 = x-x_std
                ci2 = x+x_std
                axis.fill_between( np.arange(x.shape[0]), ci1, ci2, alpha=.1)

        axis.set_yscale(scale)
        axis.set_title(title, fontsize=38)
        axis.set_xlabel("Number of trials", fontsize=38)
        axis.set_ylabel(y_label, fontsize=38)
        axis.tick_params(axis="x", labelsize=38)
        axis.tick_params(axis="y", labelsize=38)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    def generate_rank_and_regret(self):

        self.rank_per_space = {}
        self.regret_per_space = {}
        self.all_ranks = []
        self.all_regrets = []

        for _, search_space in enumerate(self.search_spaces):

            results_data = {}
            for experiment in self.experiments:
                results_data[experiment] = {}
            self.rank_per_space[search_space] = []
            self.regret_per_space[search_space] = []

            for task in self.task_list_per_space[search_space]:
                for seed in self.seeds:
                    task_seed_results = []
                    complete_results_task_seed =  True

                    for experiment in self.experiments:
                        try:
                            regret = [1-x for x in  self.results[experiment][search_space][task][seed]]
                            
                            assert len(regret) >= self.n_trials, "The task {} should have length {} in experiment {} for space {} and seed {}".format(task, self.n_trials,  experiment, search_space, seed)
                            regret = regret[:self.n_trials]
                            
                            task_seed_results.append(regret)
                        except Exception as e:
                            complete_results_task_seed = False
                            print(e)
                            print("The taks {} was probably not found for experiment {}, search space {} and seed {}".format(task, experiment, search_space, seed))
                    
                    if complete_results_task_seed:
                            rank_df = pd.DataFrame(1-np.array(task_seed_results).round(8)).rank(axis=0, ascending=False)

                            self.rank_per_space[search_space].append(rank_df.to_numpy().tolist())
                            self.regret_per_space[search_space].append(task_seed_results)
            
            self.all_ranks.extend(self.rank_per_space[search_space])
            self.all_regrets.extend(self.regret_per_space[search_space])

        return self.rank_per_space, self.regret_per_space, self.all_ranks, self.all_regrets

    def generate_plots_per_search_space(self, name = None, path = None):

        name = name if name is not None else self.name
        path = path if path is not None else self.path

        fig, axis_rank = plt.subplots(4,4, figsize=(40,32))
        fig2, axis_regret = plt.subplots(4,4, figsize=(40,32))

        for i, search_space in enumerate(self.search_spaces):
            index0 = i//4
            index1 = i%4

            if len(self.rank_per_space[search_space])>0:
                self.make_rank_and_regret_plot(self.rank_per_space[search_space], self.regret_per_space[search_space], axis_rank[index0, index1], axis_regret[index0, index1], title = "Search space No. "+search_space,)


        fig.legend(self.experiments,loc="lower center", bbox_to_anchor=(0.55, -0.05), ncol=5, fontsize=32)
        fig2.legend(self.experiments,loc="lower center", bbox_to_anchor=(0.55, -0.05), ncol=5, fontsize=32)

        fig.subplots_adjust(wspace=0.4, hspace=0.4)    
        fig2.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.tight_layout()            
        plt.draw()

        fig.savefig(path+name+"_rank.png", bbox_inches="tight")
        fig2.savefig(path+name+"_regret.png", bbox_inches="tight")


    def generate_aggregated_plots(self, name = None, path = None):

        name = name if name is not None else self.name
        path = path if path is not None else self.path


        fig, ax= plt.subplots(1,2, figsize=(20,10))
        self.make_rank_and_regret_plot(self.all_ranks, self.all_regrets, ax[0], ax[1], title = "")

        fig.legend(self.experiments,loc="lower center", bbox_to_anchor=(0.55, -0.15), ncol=5, fontsize=32)
        plt.tight_layout()
        plt.draw()
        fig.savefig(path+name+".png", bbox_inches="tight")


    def draw_cd_diagram(self,  bo_iter=50, name="Rank", path=None):

        path = path if path is not None else self.path
        df = pd.DataFrame(np.array(self.all_ranks )[:,:,bo_iter].T.tolist()).T
        df.columns = self.experiments
        df = df.stack().reset_index()
        df.columns = ["dataset_name", "classifier_name", "accuracy"]
        df.accuracy = -df.accuracy
        draw(df, path_name= path+name+".png", title=name)


    def generate_results (self, method, n_trials, new_method_name, search_spaces=None, seeds=None, *args):

        hpob_hdlr = HPOBHandler(root_dir=self.data_path, mode="v3-test")

        search_spaces = hpob_hdlr.get_search_spaces() if search_spaces is None else search_spaces
        seeds = hpob_hdlr.get_seeds() if seeds is None else seeds

        results = {}

        for search_space_id in search_spaces:

            if search_space_id not in results.keys():
                results[search_space_id] = {} 
            
            for dataset_id in hpob_hdlr.get_datasets(search_space_id):

                if dataset_id not in results[search_space_id].keys():
                    results[search_space_id][dataset_id] = {} 

                for seed in seeds:

                    if hasattr(method, "initialize"):
                        method.initialize(*args)          
                              
                    results[search_space_id][dataset_id][seed]  = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                            dataset_id = dataset_id,
                                                            seed = seed,
                                                            n_trials = n_trials )
                                                        

        with open(self.results_path+new_method_name, "w") as f:
            json.dump(results, f)


if __name__=="__main__":

    data_path = "hpob-data/"
    results_path = "results/"
    output_path = "plots/"
    name = "benchmark_plot"
    experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP"]

    benchmark_plotter  = BenchmarkPlotter(experiments=experiments, 
                                            name = name,
                                            results_path=results_path, 
                                            output_path=output_path, 
                                            data_path = data_path)

    benchmark_plotter.plot()
    benchmark_plotter.draw_cd_diagram(bo_iter=25, name="Rank@25")
    benchmark_plotter.draw_cd_diagram(bo_iter=50, name="Rank@50")
    benchmark_plotter.draw_cd_diagram(bo_iter=100, name="Rank@100")

    print("Finished")