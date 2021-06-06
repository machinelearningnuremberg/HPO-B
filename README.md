# HPO-B
HPO-B is a benchmark for assessing the performance of HPO algorithms. This repo contains the code for easing the consumption of the meta-dataset and speeding up the testing. 

## Usage

Before testing the algorithm:

* Download HPO-B from the link.
* Download the source code of this repo.
* Create a class that encapsulates the new HPO algorithm. The class should have a function called **observe_and_suggest** that will be called by **HPOBHandler object**, the class for loading the data and evaluating the algorithm.
* This function receives three parameters *X_obs, y_obs, X_pen* that represent the observed hyperparameter configurations, its response value and the configurations pending to evalute, respectively. It should return the index of the next sample to evaluate in the pending configurations (*X_pen*).

To test the algorithm:

* Create a HPOBHandler object by specifying the path and the mode.
* Three different modes are possible:
 

```python
from hpob_handler import HPOBHandler
from bo_methods import RandomSearch
import matplotlib.pyplot as plt

hpob_hdlr = HPOBHandler(root_dir="HPO-Bench/", mode="test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]

method = RandomSearch()
perf = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        trial = trial,
                                        n_iterations = 100 )


plt.plot(perf)


