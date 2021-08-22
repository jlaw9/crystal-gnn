
import argparse
import os
import sys
import copy
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm

import utils
import preprocess_crystals


def main(config_map):
    experiments = utils.get_experiments(config_map['experiments'])
    for experiment in experiments:
        # first generate the training and validation input data for the model
        exp_config_map = copy.deepcopy(config_map)
        exp_config_map['experiments'] = [experiment]
        #preprocess_crystals.main(exp_config_map)
        out_dir = utils.get_out_dir(config_map, experiment)

        run_config_file = f"{out_dir}/run.yaml"
        print(f"Writing {run_config_file}")
        with open(run_config_file, 'w') as out:
            yaml.dump(exp_config_map, out)

        # now setup the job to submit
        command = f"srun python train_model.py --config {run_config_file}"
        print(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', )
    parser.add_argument('--config-file', type=str, default="config/config.yaml",
                        help='config file to use when building dataset splits and training the model')
    # parser.add_argument('--hypo-structures', type=str, action='append',
    #                    help='path/to/hypothetical-structures.json.gz. Can specify multiple times')

    args = parser.parse_args()

    config_map = utils.load_config_file(args.config_file)
    main(config_map)
