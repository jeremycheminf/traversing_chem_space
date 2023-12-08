import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm.auto import tqdm
from active_learning.screening_one_round import active_learning_one_round
import itertools
import argparse
from ast import literal_eval
import pandas as pd


PARAMETERS = {
    "max_screen_size": [100000],
    "batch_size": [64, 32, 16],
    "architecture": ["gcn", "mlp"],
    "path_prior": ["/home/jeremy/traversing_chem_space/data/Generic/screen.csv"],
    "path_select": ["/home/jeremy/traversing_chem_space/data/Generic/select.csv"],
    "seed": list(range(10)),
    "bias": ["random", "small", "large"],
    "acquisition": [
        "random",
        "exploration",
        "exploitation",
        "dynamic",
        "dynamicbald",
        "batch_bald",
        "similarity",
        "bald",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", help="The path of the output directory", default="results"
    )
    parser.add_argument(
        "-acq",
        help="Acquisition function ('random', 'exploration', 'exploitation', 'dynamic', "
        "'batch_bald', 'similarity')",
        default="random",
    )
    parser.add_argument(
        "-bias",
        help='The level of bias ("random", "small", "large")',
        default="results",
    )
    parser.add_argument(
        "-arch", help='The neural network architecture ("gcn", "mlp")', default="mlp"
    )
    parser.add_argument(
        "-path_prior", help="The path to folder with train compounds", default=""
    )
    parser.add_argument(
        "-path_select", help="The path to folder with select compounds", default=""
    )
    # parser.add_argument('-retrain', help='Retrain the model every cycle', default='True')
    parser.add_argument(
        "-batch_size", help="How many molecules we select each cycle", default=20
    )
    parser.add_argument("-anchored", help="Anchor the weights", default="True")
    parser.add_argument(
        "-out",
        help="Path to save selected compounds",
        default="/home/jeremy/traversing_chem_space/data/Generic/out/picks.csv",
    )
    args = parser.parse_args()

    PARAMETERS["acquisition"] = [args.acq]
    PARAMETERS["bias"] = [args.bias]
    PARAMETERS["path_prior"] = [args.path_prior]
    PARAMETERS["path_select"] = [args.path_select]
    # PARAMETERS['retrain'] = [eval(args.retrain)]
    PARAMETERS["architecture"] = [args.arch]
    PARAMETERS["batch_size"] = [int(args.batch_size)]
    PARAMETERS["anchored"] = [literal_eval(args.anchored)]

    PARAMETERS = {
        "max_screen_size": [100000],
        "batch_size": [20],
        "architecture": ["gcn"],
        "path_prior": ["/home/jeremy/traversing_chem_space/data/Generic/screen.csv"],
        "path_select": ["/home/jeremy/traversing_chem_space/data/Generic/select.csv"],
        "seed": [1],
        "bias": ["small"],
        "acquisition": [
            "random",
            "exploration",
            "exploitation",
            "batch_bald",
            "similarity",
            "bald",
        ],
    }

    # LOG_FILE = f'{args.o}/{args.arch}_{args.acq}_{args.bias}_{args.batch_size}_simulation_results.csv'
    LOG_FILE = args.o
    output_file = args.out
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # PARAMETERS['acquisition'] = ['random']
    # PARAMETERS['architecture'] = ['gcn']
    # LOG_FILE = f"results/{PARAMETERS['architecture'][0]}_{PARAMETERS['acquisition'][0]}_{PARAMETERS['bias'][0]}_simulation_results.csv"

    experiments = [
        dict(zip(PARAMETERS.keys(), v)) for v in itertools.product(*PARAMETERS.values())
    ]

    for experiment in tqdm(experiments):
        # print("it is" + str(experiments))

        results, compounds = active_learning_one_round(
            acquisition_method=experiment["acquisition"],
            max_screen_size=experiment["max_screen_size"],
            batch_size=experiment["batch_size"],
            architecture=experiment["architecture"],
            seed=experiment["seed"],
            # retrain=experiment['retrain'],
            # anchored=experiment['anchored'],
            path_file_prior=experiment["path_prior"],
            path_file_select=experiment["path_select"],
            optimize_hyperparameters=False,
        )

        # Add the experimental settings to the outfile
        results["acquisition_method"] = experiment["acquisition"]
        results["architecture"] = experiment["architecture"]
        results["batch_size"] = experiment["batch_size"]
        results["seed"] = experiment["seed"]
        # col_names = ['col' + str(i) for i in np.arange(compounds.shape[0]) + 1]
        pick_df = pd.DataFrame(data=compounds)
        pick_df["acquisition_method"] = experiment["acquisition"]
        pick_df["architecture"] = experiment["architecture"]
        pick_df["batch_size"] = experiment["batch_size"]
        pick_df["seed"] = experiment["seed"]
        pick_df.to_csv(
            args.out,
            mode="a",
            index=False,
            header=False if os.path.isfile(output_file) else True,
        )

        results.to_csv(
            LOG_FILE,
            mode="a",
            index=False,
            header=False if os.path.isfile(LOG_FILE) else True,
        )
