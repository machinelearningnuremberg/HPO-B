# HPO-B
HPO-B is a benchmark for assessing the performance of black-box HPO algorithms. This repo contains the code for easing the consumption of the meta-dataset and speeding up the testing. 

## Meta-Dataset 

The meta-dataset contains evaluations of the accuracy for different search-spaces on different datasets. For more details on the meta-dataset, refer to our [paper](https://arxiv.org/pdf/2106.06257.pdf) . It is presented in three versions:

- **HPO-B-v1**: The raw benchmark of all 176 meta-datasets.
- **HPO-B-v2**: Subset of 16 meta-datasets with the most frequent search spaces.
- **HPO-B-v3**: Split of HPO-B-v2 into training, validation and testing. 

**The HPO-B benchmark meta-dataset is available  [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip)**.

**Additionally, if you wish to test on continuous search spaces, download the surrogates [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/69roMwNpG53sYoe/download/saved-surrogates.zip)**.
## Usage

Before testing a new algorithm:

* Download HPO-B data.
* Download the source code of this repo.
* Create a class that encapsulates the new HPO method. The class should have a function called `observe_and_suggest` that will be called by `HPOBHandler` object, the class for loading the data and evaluating the method.
* This function receives three parameters *X_obs, y_obs, X_pen* that represent the observed hyperparameter configurations, its response value and the configurations pending to evalute, respectively. It should return the index of the next sample to evaluate in the pending configurations (*X_pen*).

To test the algorithm:

* Create a HPOBHandler object by specifying the path to the meta-dataset and the mode.
```python
hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")
```
* 5 different modes are possible as argument:
  - **v1**: Loads HPO-B-v1
  - **v2**: Loads HPO-B-v2
  - **v3**: Loads HPO-B-v3
  - **v3-test**: Loads only the meta-test split from HPO-B-v3
  - **v3-train-augmented**: Loads all splits from HPO-B-v3, but augmenting the meta-train data with the less frequent search-spaces.
* Evaluate the new method by using the function `evaluate` of the HPOB handler. The function receives the HPO algorithm class (`method`), the search space ID, dataset ID, the seed ID and the number of optimization trials.
```python
acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        seed = "test0",
                                        n_trials = 100 )
```

* The function returns a list of the maximum accuracy achieved after every trial.
* The five valid seeds identifiers are: "test0", "test1", "test2", "test3", "test4".
* The search spaces ID and datasets ID available to evaluate can be queried by using the functions `get_search_spaces()` and `get_datasets()` of the HPOB handler.

## Usage with surrogates (continuous search-spaces)

With HPO-B, Tt is possible to perform the optimization in a continuous serch-space by using surrogates that approximate the real response function. The surrogates ware XGBoost models trained on the discrete data. If you want to perform the benchmarking on continunous search-spaces, follow these steps:

* Download this repo and the [meta-dataset](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip).
* Download the surrogate models from this [link](https://rewind.tf.uni-freiburg.de/index.php/s/69roMwNpG53sYoe/download/saved-surrogates.zip).
* Install XGBoost.
* Create a class that encapsulates the new HPO method. The class should have a function called `observe_and_suggest` that will be called by `HPOBHandler` object, the class for loading the data and evaluating the method.
* This function receives two parameters *X_obs, y_obs* that represent the observed hyperparameter configurations. It should return the index of the next sample to evaluate on the surrogate that approximates the response fuction. The valid range of the new sample is between 0 and 1 for all the components of the vector.
* Create a HPOBHandler object by specifying the path to the meta-dataset and the mode.

```python
hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/")

```
* Evaluate the new method by using the function `evaluate_continuous` of the HPOB handler. The function receives the HPO algorithm class (`method`), the search space ID, dataset ID, the seed ID and the number of optimization trials.
```python
acc = hpob_hdlr.evaluate_continuous(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        seed = "test0",
                                        n_trials = 100 )
```

## Requirements
* Python 3.7
* botorch (optional for running advanced examples)
* pyGPGO (optional for running advanced examples)
* XGBoost (option for using the surrogates for a continuous search space)

## Basic example
```python
from hpob_handler import HPOBHandler
from methods.random_search import RandomSearch
import matplotlib.pyplot as plt

#Alternatively, for a continuous search space: 
#hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/")
hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]

method = RandomSearch()

#Alternatively, for a continuous search space: acc = hpob_hdlr.evaluate_continuous(...)
acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        seed = "test0",
                                        n_trials = 100 )


plt.plot(acc)
```
## Advanced examples

For more advanced examples on how to use more methods and fully evaluate a search space using all the seeds, refer to the files `example_botorch.py` or `example_pygpgo.py`.


## Cite us
```
@misc{arango2021hpob,
      title={HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML}, 
      author={Sebastian Pineda Arango and Hadi S. Jomaa and Martin Wistuba and Josif Grabocka},
      year={2021},
      eprint={2106.06257},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
