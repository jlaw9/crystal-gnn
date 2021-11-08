
import argparse
import os
import copy
import yaml
import subprocess

from src import utils
#import preprocess_crystals_split_all_data as preprocess_crystals
import preprocess_crystals as preprocess_crystals


def main(config_map, 
         valid_split_all_data=None, 
         submit=False, 
         forced=False):
    experiments = utils.get_experiments(config_map['experiments'])
    for experiment in experiments:
        exp_config_map = copy.deepcopy(config_map)
        exp_config_map['experiments'] = [experiment]
        # first generate the training and validation input data for the model
        if valid_split_all_data:
            out_dirs = preprocess_crystals.setup_experiment(config_map, 
                             experiment, 
                             valid_split_all_data=valid_split_all_data, 
                             forced=forced)

            for out_dir in out_dirs:
                run_experiment(config_map, 
                               out_dir, 
                               exp_config_map, 
                               valid_split_all_data=valid_split_all_data, 
                               submit=submit, 
                               forced=forced)
        else:
            # make sure the inputs are setup
            out_dir = preprocess_crystals.setup_experiment(config_map, experiment, forced=forced)
            #out_dir = utils.get_out_dir(config_map, experiment)

            # now create and submit the job
            run_experiment(config_map, 
                           out_dir, 
                           exp_config_map, 
                           submit=submit, 
                           forced=forced)


def run_experiment(config_map, 
                   out_dir,
                   exp_config_map, 
                   valid_split_all_data=None, 
                   submit=False, 
                   forced=False):
    params = utils.check_default_hyperparams(config_map['hyperparameters'])
    model_dir = utils.get_hyperparam_dir(out_dir, params)
    os.makedirs(model_dir, exist_ok=True)

    model_file = f"{model_dir}/best_model.hdf5"
    if not forced and os.path.isfile(model_file):
        print(f"\t{model_file} already exists. TODO use --foced to re-run")
        return

    run_config_file = f"{model_dir}/run.yaml"
    print(f"Writing {run_config_file}")
    with open(run_config_file, 'w') as out:
        yaml.dump(exp_config_map, out)

    # now setup the job to submit
    command = f"python -u train_model.py --config {run_config_file} --exp-dir {out_dir}"
    #print(command)

    out_file = f"{model_dir}/submit.sh"
    write_submit_script(out_file, 
            command, 
            name=os.path.basename(os.path.dirname(model_dir)),
            log_file=f"{model_dir}/log.out",
            err_file=f"{model_dir}/err.out",
            email_username=os.environ['USER'],
            )

    # now submit to the queue
    if submit:
        submit_command = f"sbatch {out_file}"
        print(submit_command + '\n')
        subprocess.check_call(submit_command, shell=True)

def write_submit_script(out_file,
                        command,
                        name="test-crystals",
                        log_file="",
                        err_file="",
                        email_username=os.environ['USER']):
    out_str = f"""#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=10:00:00
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --mail-user={email_username}@nrel.gov
#SBATCH --mail-type=END

source ~/.bashrc_conda
module load cudnn/8.1.1/cuda-11.2
#conda activate crystals
conda activate ~/.conda-envs/rlmol

echo "Job started at `date`"
srun {command}
echo "Job ended at `date`"
"""

    print(f"Writing {out_file}")
    with open(out_file, 'w') as out:
        out.write(out_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', )
    parser.add_argument('--config-file', type=str, default="config/config.yaml",
                        help='config file to use when building dataset splits and training the model')
    parser.add_argument('--valid-split-all-data', type=int, 
                        help='Make train/valid splits across the entire dataset. ' + \
                             'Specify the # K-fold splits to make ' + \
                             'so that all structures are left out of training at some point')
    parser.add_argument('--submit', action='store_true', default=False,
                        help='Submit to the HPC envrionment using sbatch')
    # parser.add_argument('--hypo-structures', type=str, action='append',
    #                    help='path/to/hypothetical-structures.json.gz. Can specify multiple times')

    args = parser.parse_args()

    config_map = utils.load_config_file(args.config_file)
    main(config_map, 
         submit=args.submit, 
         valid_split_all_data=args.valid_split_all_data)
