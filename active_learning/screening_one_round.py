"""

This script contains the one loop of active learning experiment.
It also removes checking that selected compounds are in traning set for multiple loops
This script can be called on a set of compounds and provide compounds to push to synthesis
or extra scoring systems

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

import pandas as pd
import numpy as np
from active_learning.nn import Ensemble
from active_learning.data_prep import MasterDataset, MasterDatasetPath
from active_learning.data_handler import Handler
from active_learning.utils import Evaluate, to_torch_dataloader
from active_learning.acquisition import Acquisition, logits_to_pred
from tqdm.auto import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
from math import ceil



INFERENCE_BATCH_SIZE = 512
TRAINING_BATCH_SIZE = 64
NUM_WORKERS = 4

def active_learning_one_round(acquisition_method: str = 'exploration', max_screen_size: int = None,
                    batch_size: int = 16, architecture: str = 'gcn', seed: int = 0, bias: str = 'random',
                    optimize_hyperparameters: bool = False, ensemble_size: int = 10, retrain: bool = True,
                    anchored: bool = True, path_file_prior: str = None, path_file_select: str = None) -> pd.DataFrame:
    """
    :param n_start: number of molecules to start out with
    :param acquisition_method: acquisition method, as defined in active_learning.acquisition
    :param max_screen_size: we stop when this number of molecules has been screened
    :param batch_size: number of molecules to add every cycle
    :param architecture: 'gcn' or 'mlp'
    :param seed: int 1-20
    :param bias: 'random', 'small', 'large'
    :param optimize_hyperparameters: Bool
    :param ensemble_size: number of models in the ensemble, default is 10
    :return: dataframe with results
    """

    # Load the datasets
    representation = 'ecfp' if architecture == 'mlp' else 'graph'
    #print(f'the path is {path_file_prior}')
    ds_train = MasterDatasetPath('train', representation=representation, path=path_file_prior,select = False)
    ds_select = MasterDatasetPath('select', representation=representation, path=path_file_select,select = True)

    # Initiate evaluation trackers
    eval_screen, eval_train = Evaluate(), Evaluate()
    #handler = Handler(n_start=n_start, seed=seed, bias=bias, dataset=dataset)

    # Define some variables
    
    max_screen_size = len(ds_select) if max_screen_size is None else max_screen_size

    n_cycles = 1
    # exploration_factor = 1 / lambd^x. To achieve a factor of 1 at the last cycle: lambd = 1 / nth root of 2
    lambd = 1 / (2 ** (1/n_cycles))

    ACQ = Acquisition(method=acquisition_method, seed=seed, lambd=lambd)

        # Get the train and screen data for this cycle
    x_train, y_train, smiles_train = ds_train.all()
    x_screen, y_screen, smiles_screen = ds_select.all()

    # Get class weight to build a weighted random sampler to balance out this data
    class_weights = [1 - sum((y_train == 0) * 1) / len(y_train), 1 - sum((y_train == 1) * 1) / len(y_train)]
    weights = [class_weights[i] for i in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)

    # Get the screen and train + balanced train loaders
    train_loader = to_torch_dataloader(x_train, y_train,
                                       batch_size=INFERENCE_BATCH_SIZE,
                                       num_workers=NUM_WORKERS,
                                       shuffle=False, pin_memory=True)

    train_loader_balanced = to_torch_dataloader(x_train, y_train,
                                                batch_size=TRAINING_BATCH_SIZE,
                                                sampler=sampler,
                                                num_workers=NUM_WORKERS,
                                                shuffle=False, pin_memory=True)

    screen_loader = to_torch_dataloader(x_screen, y_screen,
                                        batch_size=INFERENCE_BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False, pin_memory=True)

    # Initiate and train the model (optimize if specified)
    print("Training model")

    model = Ensemble(seed=seed, ensemble_size=ensemble_size, architecture=architecture, anchored=anchored)
    if optimize_hyperparameters:
        model.optimize_hyperparameters(x_train, y_train)
    model.train(train_loader_balanced, verbose=False)

    # Do inference of the train/test/screen data
    print("Train/test/screen inference")
    train_logits_N_K_C = model.predict(train_loader)
    eval_train.eval(train_logits_N_K_C, y_train)

    screen_logits_N_K_C = model.predict(screen_loader)
    eval_screen.eval(screen_logits_N_K_C, y_screen)

    # Select the molecules to add for the next cycle
    print("Sample acquisition")
    picks = ACQ.acquire(screen_logits_N_K_C, smiles_screen, hits=smiles_train, n=batch_size)
    #handler.add(picks)

    # Add all results to a dataframe
    train_results = eval_train.to_dataframe("train_")
    screen_results = eval_screen.to_dataframe('screen_')
    results = pd.concat([train_results, screen_results], axis=1)

    return results, picks
