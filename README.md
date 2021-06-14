# HPO-B
HPO-B is a benchmark for assessing the performance of HPO algorithms. This repo contains the code for easing the consumption of the meta-dataset and speeding up the testing. 

## Meta-Dataset 

The meta-dataset contains evaluations of the accuracy for different search-spaces on different datasets. For more details on the meta-dataset, refer to our [paper](https://arxiv.org/pdf/2106.06257.pdf) . It is presented in three versions:

- **HPO-B-v1**: The raw benchmark of all 176 meta-datasets.
- **HPO-B-v2**: Subset of 16 meta-datasets with the most frequent search spaces.
- **HPO-B-v3**: Split of HPO-B-v2 into training, validation and testing. 

**The HPO-B benchmark meta-dataset is available  [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip)**

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

## Requirements
* Python 3.7
* botorch (optional for running advanced examples)
* pyGPGO (optional for running advanced examples)

## Basic example
```python
from hpob_handler import HPOBHandler
from methods.random_search import RandomSearch
import matplotlib.pyplot as plt

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]

method = RandomSearch()
acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        seed = "test0",
                                        n_trials = 100 )


plt.plot(acc)
```
## Advanced examples

For more advanced examples on how to use more methods and fully evaluate a search space using all the seeds, refer to the files `example_botorch.py` or `example_pygpgo.py`.


## Cite us

@misc{arango2021hpob,
      title={HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML}, 
      author={Sebastian Pineda Arango and Hadi S. Jomaa and Martin Wistuba and Josif Grabocka},
      year={2021},
      eprint={2106.06257},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
