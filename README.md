# HPO-B
HPO-B is a benchmark for assessing the performance of HPO algorithms. This repo contains the code for easing the consumption of the meta-dataset and speeding up the testing. 

## Meta-Dataset 

The meta-dataset contains evaluations of the accuracy for different search-spaces on different datasets. For more details on the meta-dataset, refer to our paper (\*). It is presented in three versions:

- **HPO-B-v1**: The raw benchmark of all 176 meta-datasets;
- **HPO-B-v2**: Subset of 16 meta-datasets with the most frequent search spaces;
- **HPO-B-v3**: Split of HPO-B-v2 into training, validation and testing. 

## Usage

Before testing the algorithm:

* Download HPO-B data.
* Download the source code of this repo.
* Create a class that encapsulates the new HPO algorithm. The class should have a function called **observe_and_suggest** that will be called by **HPOBHandler object**, the class for loading the data and evaluating the algorithm.
* This function receives three parameters *X_obs, y_obs, X_pen* that represent the observed hyperparameter configurations, its response value and the configurations pending to evalute, respectively. It should return the index of the next sample to evaluate in the pending configurations (*X_pen*).

To test the algorithm:

* Create a HPOBHandler object by specifying the path and the mode.
* 5 different modes are possible as argument:
  - **v1**: Loads HPO-B-v1
  - **v2**: Loads HPO-B-v2
  - **v3**: Loads HPO-B-v3
  - **v3-test**: Loads only the meta-test split from HPO-B-v3
  - **v3-train-augmented**: Loads all splits from HPO-B-v3, but with the augmenting the meta-train data with the less frequent search-spaces.

```python
from hpob_handler import HPOBHandler
from methods.random_search import RandomSearch
import matplotlib.pyplot as plt

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]

method = RandomSearch()
perf = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        trial = trial,
                                        n_iterations = 100 )


plt.plot(perf)


