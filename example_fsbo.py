import matplotlib.pyplot as plt
import numpy as np
from hpob_handler import HPOBHandler
from methods.fsbo.fsbo_modules import DeepKernelGP
import os 

randomInitializer = np.random.RandomState(0) 
hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")
search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[0]
seed  = "test0"
n_trials = 20
load_model = True #if load_Model == False the DeepKernel is randomly initialized 


dim = hpob_hdlr.get_search_space_dim(search_space_id)
torch_seed = randomInitializer.randint(0,100000)

rootdir     = os.path.dirname(os.path.realpath(__file__))
log_dir     = os.path.join(rootdir,"methods", "fsbo","logs",f"seed-{seed}")
os.makedirs(log_dir,exist_ok=True)
log_dir = os.path.join(log_dir, f"{dataset_id}.txt")

#loads pretran model from the checkpoint "FSBO",
checkpoint = os.path.join(rootdir,"methods","fsbo","checkpoints","FSBO", f"{search_space_id}")

#define the DeepKernelGP as HPO method
method = DeepKernelGP(dim, log_dir, torch_seed, epochs = 1000, load_model = load_model, checkpoint = checkpoint, verbose = True)

#evaluate the HPO method
acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        seed = seed,
                                        n_trials = n_trials )

        

plt.plot(np.array(acc))
plt.show()
