import argparse
import os, sys
import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import tensorflow_addons as tfa  # need to import so the model is loaded correctly
from tqdm import tqdm
import monotonic
mtime = monotonic.time.time
t0 = mtime()

import nfp
sys.path.insert(0, '')
from src.nfp_extensions import RBFExpansion, CifPreprocessor
from src import utils


# now apply the lattice
def apply_cubic(struc):
    struc.lattice = struc.lattice.cubic(1.0)
    return struc


def main(config_map, exp_dir=None, eval_valid=False):
    experiments = utils.get_experiments(config_map['experiments'])
    for experiment in experiments:
        print("\n" + '-'*50)
        eval_experiment(experiment, exp_dir=exp_dir, eval_valid=eval_valid)


loaded_datasets = {}
def eval_experiment(experiment, exp_dir=None, eval_valid=False):
    if exp_dir is None:
        exp_dir = utils.get_out_dir(config_map, experiment)
    params = utils.check_default_hyperparams(config_map['hyperparameters'])
    # now get the directory in which to put this model file (distinguished by hyperparameters)
    model_dir = utils.get_hyperparam_dir(exp_dir, params)
    print(model_dir)

    # Initialize the preprocessor class.
    preprocessor = CifPreprocessor(num_neighbors=12)
    preprocessor.from_json(f'{exp_dir}/preprocessor.json')
        
    model = tf.keras.models.load_model(
        f'{model_dir}/best_model.hdf5',
        custom_objects={**nfp.custom_objects, **{'RBFExpansion': RBFExpansion}})

    test_file = f"{exp_dir}/test.csv.gz"
    if eval_valid:
        test_file = f"{exp_dir}/valid.csv.gz"
    print(f"Reading {test_file}")
    test = pd.read_csv(test_file, dtype={'id': str})
    print(test.head(2))

    # TODO use the same code here as preprocess_crystals.py
    #icsd_df, hypo_df = preprocess_crystals.load_datasets(config_map, experiment) 

    # first check if there are relaxed and unrelaxed datasets.
    # If so, just evaluate the unrelaxed structures
    relaxed_status = [config_map['datasets'][d].get('relaxed') for d in experiment['datasets']]
    skip_relaxed = True if True in relaxed_status and False in relaxed_status else False

    # load the true values to evaluate against
    hypo_structures = {} 
    icsd_structures = {} 
    for dataset_name in experiment['datasets']:
        dataset = config_map['datasets'][dataset_name]
        if dataset['hypothetical'] is True:
            if skip_relaxed and dataset['relaxed'] is True:
                print(f"Skipping '{dataset_name}' since both relaxed and unrelaxed are present")
                # only evaluate the unrelaxed if both are present
                continue
        # check if this dataset has already been loaded
        structures = loaded_datasets.get(dataset_name)
        if structures is None:
            structures = utils.load_structures_from_json(dataset['structures_file'])
            loaded_datasets[dataset_name] = structures

        set_vol_to = experiment.get('set_vol_to')
        if set_vol_to:
            print(f"Setting volume to {set_vol_to}")
            def set_vol(strc, vol):
                strc.scale_lattice(float(vol))
                return strc
            structures = {s_id: set_vol(s, float(set_vol_to)) for s_id, s in structures.items()}

        if dataset['hypothetical'] is True:
            hypo_lattice = experiment['eval_settings'].get('hypo_lattice', 'orig')
            if hypo_lattice == 'cubic':
                print(f"\tconverting hypothetical structures to '{hypo_lattice}'")
                for s_id in set(structures.keys()) & set(test.id.values):
                    structures[s_id] = apply_cubic(structures[s_id])

            if experiment.get('set_vol_to_relaxed') and dataset.get('relaxed') is False:
                # also load the relaxed structures, and set the volume of the unrelaxed structures to match the relaxed
                set_vol_to_relaxed = config_map['datasets'][dataset['relaxed_dataset']]['structures_file']
                print("Setting the volume of the unrelaxed structures to match their corresponding relaxed structures")
                rel_structures = utils.load_structures_from_json(set_vol_to_relaxed)
                for s_id in set(structures.keys()) & set(test.id.values):
                    rel_struc = rel_structures.get(s_id)
                    structures[s_id].scale_lattice(rel_struc.volume)

            print(f"Reading {dataset['relaxed_energies']}")
            hypo_relaxed = pd.read_csv(dataset['relaxed_energies'], dtype={'id': str})
            print(hypo_relaxed.head())
            # allow for multiple hypothetical datasets
            hypo_structures.update(structures)
        else:
            print(f"Reading {dataset['energies']}")
            icsd_df = pd.read_csv(dataset['energies'])
            print(icsd_df.head())
            # allow for multiple icsd datasets
            icsd_structures.update(structures)

    test_strcs = {} 
    if len(icsd_structures) > 0:
        test_icsd = test[test.id.isin(icsd_df.id)].reset_index(drop=True)
        test_strcs.update({s: icsd_structures[s] for s in test_icsd.id})
        if len(hypo_structures) > 0:
            test_hypo = test[~test.id.isin(icsd_df.id)].reset_index(drop=True)
            test_strcs.update({s: hypo_structures[s] for s in test_hypo.id})
    elif len(icsd_structures) == 0:
        test_hypo = test.reset_index(drop=True)
        test_strcs.update({s: hypo_structures[s] for s in test_hypo.id})
    #print(test_icsd)
    #print(test_hypo)

    if len(icsd_structures) > 0:
        # transform the structures to a different lattice if specified
        icsd_lattice = experiment['eval_settings'].get('icsd_lattice', 'orig')

        if icsd_lattice == 'cubic':
            print(f"\tconverting ICSD structures to '{icsd_lattice}'")
            for s in test_icsd.id:
                test_strcs[s] = apply_cubic(test_strcs[s])

        icsd_dataset = tf.data.Dataset.from_generator(
            lambda: (preprocessor.construct_feature_matrices(test_strcs[s], train=False)
                     for s in tqdm(test_icsd.id)),
            output_types=preprocessor.output_types,
            output_shapes=preprocessor.output_shapes)\
            .padded_batch(batch_size=32,
                          padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                          padding_values=preprocessor.padding_values)

        predictions_icsd = model.predict(icsd_dataset)
        test_icsd['predicted_energyperatom'] = predictions_icsd

    if len(hypo_structures) > 0:
        hypo_dataset = tf.data.Dataset.from_generator(
            lambda: (preprocessor.construct_feature_matrices(test_strcs[s], train=False)
                     for s in tqdm(test_hypo.id)),
            output_types=preprocessor.output_types,
            output_shapes=preprocessor.output_shapes)\
            .padded_batch(batch_size=32,
                          padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                          padding_values=preprocessor.padding_values)


        predictions_hypo = model.predict(hypo_dataset)
        test_hypo['predicted_energyperatom'] = predictions_hypo

    df = pd.DataFrame()
    if len(icsd_structures) > 0:
        df = df.append(test_icsd)
    if len(hypo_structures) > 0:
        df = df.append(test_hypo)
    df.to_csv(f'{model_dir}/predicted_test.csv', index=False)

    ##MAE and RMSE##
    realVals = df.energyperatom
    predictedVals = df.predicted_energyperatom
    rmse = mean_squared_error(realVals, predictedVals, squared = False)

    out_str = ""
    out_str += f'Test MAE: {(df.energyperatom - df.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom\n'
    out_str += f"Test RMSE: {rmse}\n"
    if len(icsd_structures) > 0:
        out_str += f'ICSD MAE: {(test_icsd.energyperatom - test_icsd.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom\n'
    if len(hypo_structures) > 0:
        out_str += f'Hypo MAE: {(test_hypo.energyperatom - test_hypo.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom\n'
    print(out_str)
    with open(f'{model_dir}/mae_test.txt','w') as out:
        out.write(out_str)

    elapsed = mtime()-t0
    f_time = open(f'{model_dir}/time_test.txt', 'w')
    print("elaspsed_time:", elapsed, file=f_time)
    f_time.close()


def plot_train_and_parity(pred_df, icsd_df, out_file):
    """ Plot the training curves and a parity plot

    :param pred_df: pandas dataframe containing the energy per atom
        as well as the predicted energy per atom
    :param icsd_df: pandas dataframe containing the IDs of ICSD structures
    """
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4), aspect='equal')

    plot_train_curves(log_df, ax=ax1)
    plot_parity(pred_df, icsd_df, ax=ax2)


def plot_parity(pred_df, icsd_df, ax=None, out_file=None):
    icsd_pred_df = pred_df[pred_df.id.isin(icsd_df.id)]
    hypo_pred_df = pred_df[~pred_df.id.isin(icsd_df.id)]

    if ax is None:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, aspect='equal')
    ax.plot(icsd_pred_df.energyperatom, icsd_pred_df.predicted_energyperatom, '.', ms=2, c='b', label='ICSD')
    ax.plot(hypo_pred_df.energyperatom, hypo_pred_df.predicted_energyperatom, '.', ms=2, c='r', label='Hypothetical')
    ax.plot([-9, -1], [-9, -1], '--', color='.8', zorder=0)
    ax.set_xticks([-1, -3, -5, -7, -9])
    ax.set_yticks([-1, -3, -5, -7, -9])
    ax.legend(loc='upper left', frameon=False, prop={'size':10}, markerscale=6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(direction='in', length=5)
    #plt.ylabel('Energy per atom,\npredicted (eV/atom)', fontsize=12)
    #plt.xlabel('Energy per atom,\nDFT (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Total Energy (eV/atom)', fontsize=14)
    plt.xlabel('DFT Total Energy (eV/atom)', fontsize=14)
    plt.tight_layout()

    ax.text(1, 0.025, f'Test MAE: {(test.energyperatom - test.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom',
            ha='right', va='bottom', transform=ax.transAxes, fontsize=14)

    sns.despine(trim=False)
    if out_file is not None:
        print("writing {out_file}")
        plt.savefig(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', )
    parser.add_argument('--config-file', type=str, default="config/config.yaml",
                        help='config file to use when building dataset splits and training the model')
    parser.add_argument('--exp-dir', type=str,
                        help='If the experiment dir is different than what the script would extract from the config file, then set that here.')
    parser.add_argument('--eval-valid', action='store_true', default=False,
                        help='Evaluate on the validation set (i.e., valid.csv.gz) rather than the test set (i.e., test.csv.gz)')
    # parser.add_argument('--hypo-structures', type=str, action='append',
    #                    help='path/to/hypothetical-structures.json.gz. Can specify multiple times')

    args = parser.parse_args()

    config_map = utils.load_config_file(args.config_file)
    main(config_map, exp_dir=args.exp_dir, eval_valid=args.eval_valid)

