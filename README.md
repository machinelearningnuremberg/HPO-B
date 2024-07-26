# HPO-B
HPO-B is a benchmark for assessing the performance of black-box HPO algorithms. This repo contains the code for easing the consumption of the meta-dataset and speeding up the testing. 

## Meta-Dataset 

The meta-dataset contains evaluations of the accuracy for different search-spaces on different datasets. For more details on the meta-dataset, refer to our [paper](https://arxiv.org/pdf/2106.06257.pdf). It is presented in three versions:

- **HPO-B-v1**: The raw benchmark of all 176 meta-datasets.
- **HPO-B-v2**: Subset of 16 meta-datasets with the most frequent search spaces.
- **HPO-B-v3**: Split of HPO-B-v2 into training, validation and testing. 

In al our settings the response function is to be **maximized**.

**The HPO-B benchmark meta-dataset is available  [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip)**.

**Additionally, if you wish to test on continuous search spaces, download the surrogates [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip)**.
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
* Download the surrogate models from this [link](https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip). Every surrogate is an XGBoost model, whose name follows the pattern: "surrogate-[search_space_id]+[task_id].json".
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
* XGBoost (optional for using the surrogates on a continuous search space)
* networkx (optional for using the `BenchmarkPlotter`
* matpotlib 3.4.2
* cd_diagram.py: donwload file from [HERE](https://github.com/hfawaz/cd-diagram/blob/master/main.py) and put in the main folder.
* ujson (optinonal for faster loading of json files)

## Basic example

Now we will explain how to create a class wrapper for a new algorithm and how to evaluate it on HPO-B.

### 1. HPO algorithm class

For creating a new method, we firstly create a python class, with a constructor and class method called `observe_and_suggest`. In general, the function receives three arguments: 

* `X_obs`: a list of list with the observed configurations. For instance, two observed configurations with three hyperparameters would look like `[[1,2,3],[4,5,6]]`.
* `y_obs`: a list of list with the observed responses. For instnace, tthe responses for two configurations looks like `[[0.9], [0.7]]`.
* `X_pen`: a list of list with the pending configurations to evaluate. For instance, two pending configurations with three hyperparameters would look like `[[1,2,3],[4,5,6]]`.
Alternatively, X_pen could be `None`, in case of using the continuous search space. Therefore, this functionality should be implemented when using the XGBoost surrogates.


```python
class RandomSearch:

    def __init__(self):

      print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
      
      if X_pen not None:
        # code for discrete search space
        
      else:
        # code for continuous search space
```
### 2. `observe_and_suggest` method 

As indicated previously, `observe_and_suggest` have two possible funcional modes, depending on whether `X_pen` is specified or not. In case it is specified, it is assumed to be using the discrete benchmark, therefore, it should return the index of the next configuration to evaluate from the list of configurations specified by `X_pen`. For a random search implementation, this means to select randomly a value between 0 and the number of pending configurations:


```python
size_pending_eval = len(X_pen)
idx = random.randint(0, size_pending_eval-1)
return idx

```
When it is not specified, it is assumed to use the continuous benchmark, therefore the output should be a list (sample) with the same dimensionality as a the observed samples `X_obs`. Moreover, given the characteristics of the benchmark, the values should be between 0 and 1.

```python
dim = len(X_obs[0])
bounds = tuple([(0,1) for i in range(dim)])
x_new = np.array([random.uniform(lower, upper) for upper, lower in bounds]).reshape(-1, dim)
return x_new
```

Now, we can summarize our new HPO algorithm class (Random Search in this example) as:

```python
import random
import numpy as np

class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):

        if X_pen is not None:
            size_pending_eval = len(X_pen)
            idx = random.randint(0, size_pending_eval-1)
            return idx

        else:
            dim = len(X_obs[0])
            bounds = tuple([(0,1) for i in range(dim)])
            x_new = np.array([random.uniform(lower, upper) for upper, lower in bounds]).reshape(-1, dim)

            return x_new
```

### 3. Use the new class with the `HPOBHandler` 

Once we created the class, we can use it on the HPOBHandler, which will call our method `observe_and_suggest` to benchmark our new algorithm. It returns a list with the incumbent's accuracy at every HPO iteration.

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


## Meta-dataset format

As described in the paper, the meta-dataset follows a JavaScript Object Notation (JSON) to encapsulate the evaluations. In Python, this corresponds to nested dictionaries, where the first level key corresponds to the **search space ID**, the second level key contains the **dataset ID**, and finally the last level contains the list of hyperparameter configurations (**X**) and its response (**y**).


```python
meta_dataset = { "search_space_id_1" : { "dataset_id_1": {"X": [[1,1], [0,2]],
                                                  "y": [[0.9], [0.1]]},
                               { "dataset_id_2": ... },
                 "search_space_id_2" : ...
                                
                }
```
## Extending the benchmark

In the folder `meta-dataset-creation`, we provide the code of the full benchmark extraction (`data_extraction.py`) and an example of preprocessing for a single search space. Based on this, it is possible for any user to add new search spaces (called flows in OpenML) Also, it is possible to download existing search spaces and extend it with new evaluations.

## Generating benchmark plots

If you want to compare a new model to the results in our [paper](https://arxiv.org/pdf/2106.06257.pdf), you should generate a list of incumbents accuracy for every search-space, dataset and seed combination and wrap the results in a JSON file with the following structure:

```python
new_model_results = { "search_space_id_1" : { "dataset_id_1": {"test0": [0.85, 0.9,...],
                                                                "test1": ...}
                                            { "dataset_id_2": ... },
                      "search_space_id_2" : ...
                                
                    }
```

Afterwards, you can use the `BenchmarkPlotter` class from `benchmark_plot.py`. A complete example can be found in `example_benchmark.py`.

## Update (June 28 2022)

A refactored implementation of FSBO, the best performing model according to our benchmark, is avaialble in this [repo](https://github.com/releaunifreiburg/FSBO). We also included an evaluation on continuous search spaces, which was not present in the original paper. During the evaluation of these results, we realied that the minimum and maximum values for tha tasks on the space 5970 were wrong. The fixed values are already provided in the updated link for the surrogates. We include the evaluations for the continuous search spaces under the folder `results`, with names `RS-C.json, GP-C.json, DGP-C.json`.

## Update (September 30 2022)

We added `hpob-data/meta-dataset-descriptors.json`, a JSON file containing the descriptions of the original variable names and ranges per search space (first level in the file). Moreover, we specify which variables have log transformation in the meta-dataset.

## Update (June 13 2023)

We remove `cd_diagram.py` due to license. We recommend to download the file from the original [author](https://github.com/hfawaz/cd-diagram/main.py) and rename it as `cd_diagram.py`.

## Cite us
```

@article{pineda2021hpob,
  author    = {Sebastian Pineda{-}Arango and
               Hadi S. Jomaa and
               Martin Wistuba and
               Josif Grabocka},
  title     = {{HPO-B:} {A} Large-Scale Reproducible Benchmark for Black-Box {HPO}
               based on OpenML},
  journal   = {Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  year      = {2021}
}

```
