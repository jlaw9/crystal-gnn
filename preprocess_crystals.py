import argparse
import os
import sys
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
#tqdm.pandas()
import tensorflow as tf
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
import pickle

import nfp

import utils
from nfp_extensions import CifPreprocessor


def load_icsd(icsd_energies_file, icsd_structures_file):
    icsd_df = pd.read_csv(icsd_energies_file)
    structures = utils.load_structures_from_json(icsd_structures_file)

    def get_strc(strc_id):
        struc = structures.get(strc_id)
        #struc.lattice = struc.lattice.cubic(1.0)                  # cubic lattice with a=b=c=1
        #struc.lattice = struc.lattice.tetragonal(1.0,3.0)         # tetragonal lattice with a=b=1,c=3
        #struc.lattice = struc.lattice.orthorhombic(1.0,2.0,3.0)   # orthorhombic lattice with a=1,b=2,c=3
        return struc

    icsd_df['crystal'] = icsd_df.id.progress_apply(get_strc)

    return icsd_df


def load_hypothetical_structures(
        relaxed_energies_file, structures_file):
    hypo_df = pd.read_csv(relaxed_energies_file)
    structures = utils.load_structures_from_json(structures_file)

    def get_strc(strc_id):
        struc = structures.get(strc_id)
        return struc

    hypo_df['crystal'] = hypo_df.id.progress_apply(get_strc)

    return hypo_df


def random_split_df(df, test_size_percentage=.05, random_state=None):
    strc_ids = df.id.unique()
    train, valid, test = random_split(
        strc_ids, test_size_percentage=test_size_percentage, random_state=random_state)
    train_df = df[df.id.isin(train)]
    valid_df = df[df.id.isin(valid)]
    test_df = df[df.id.isin(test)]
    return train_df, valid_df, test_df


def random_split(structure_ids, test_size_percentage=.05, random_state=None):
    test_size = len(structure_ids) * test_size_percentage
    print(f"\tsplitting {len(structure_ids)} structures using test_size: {test_size_percentage} ({test_size})")
    train, test  = train_test_split(structure_ids, test_size=test_size, random_state=random_state)
    train, valid = train_test_split(train, test_size=test_size, random_state=random_state)
    return train, valid, test


def leave_out_comp(df, random_state=None):
    # Split the hypothetical data into training and test sets, such that test/valid sets have one composition per comp_type
    valid = df.groupby("comp_type").sample(n=1, random_state=random_state)
    train = df[~df.composition.isin(valid.composition)]
    test  = train.groupby("comp_type").sample(n=1, random_state=random_state)
    train = train[~train.composition.isin(test.composition)]
    return train, valid, test


def main(config_map):
    for experiment in config_map['experiments']:
        hypo_df = pd.DataFrame()
        icsd_df = pd.DataFrame()
        dataset_names = '-'.join(experiment['datasets'])
        for dataset_name in experiment['datasets']:
            dataset = config_map['datasets'][dataset_name]
            if dataset['hypothetical'] is True:
                curr_hypo_df = load_hypothetical_structures(
                    dataset['relaxed_energies'], dataset['structures_file'])
                hypo_df = pd.concat([hypo_df, curr_hypo_df])
            else:
                # treat this as an ICSD dataset
                curr_icsd_df = load_icsd(
                    dataset['energies'], dataset['structures_file'])
                icsd_df = pd.concat([icsd_df, curr_icsd_df])

        eval_settings = experiment['eval_settings']
        random_state = eval_settings.get('seed')
        if random_state is not None:
            print(f"Using random_state: {random_state}")

        # Split the icsd data into training and test sets
        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()
        eval_strs = []
        for df, dataset_type in [(icsd_df, 'icsd'), (hypo_df, 'hypo')]:
            if len(df) == 0:
                continue
            eval_type = eval_settings.get(dataset_type, 'random_subset')
            print(f"Making splits for {dataset_type} using '{eval_type}'")
            if eval_type == "random_subset":
                eval_str = f"{dataset_type}_{eval_type}"
                train, valid, test = random_split_df(df, random_state=random_state)
                #icsd_train, icsd_valid, icsd_test = random_split(icsd_ids, random_state=random_state)
            elif eval_type == "leave_out_comp":
                eval_str = f"{dataset_type}_loc"
                train, valid, test = leave_out_comp(df, random_state=random_state)
            elif eval_type == "leave_out_comp_minus_one":
                eval_str = f"{dataset_type}_loc-1"
                sys.exit(f"{eval_type} not yet implemented. Quitting")

            print(f"# train: {len(train)}, # valid: {len(valid)}, # test: {len(test)}")
            train_df = pd.concat([train_df, train])
            valid_df = pd.concat([valid_df, valid])
            test_df = pd.concat([test_df, test])
            eval_strs.append(eval_str)

        full_eval_str = "-".join(eval_strs)

    # Initialize the preprocessor class.
    preprocessor = CifPreprocessor(num_neighbors=12)

    def inputs_generator(df, train=True):
        """ This code encodes the preprocessor output (and prediction target) in a 
        tf.Example format that we can use with the tfrecords file format. This just
        allows a more straightforward integration with the tf.data API, and allows us
        to iterate over the entire training set to assign site tokens.
        """
        for i, row in tqdm(df.iterrows(), total=len(df)):
            input_dict = preprocessor.construct_feature_matrices(row.crystal, train=train)
            input_dict['energyperatom'] = float(row.energyperatom)
            #input_dict['a'] = float(row.a)
            #input_dict['b'] = float(row.b)
            #input_dict['c'] = float(row.c)
            #input_dict['alpha'] = float(row.alpha)
            #input_dict['beta'] = float(row.beta)
            #input_dict['gamma'] = float(row.gamma)
            #input_dict['vol_atom'] = float(row.vol_atom)
           
            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    # Process the training data, and write the resulting outputs to a tfrecord file
    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(train, train=True),
        output_types=tf.string, output_shapes=())

    # structure of outputs:
    # <output_dir>/
    #   <dataset_name>/
    #     <evaluation>_<seed>/
    #       train/valid/test splits
    #       <hyperparameters>/
    #         trained model
    out_dir = os.path.join(config_map['output_dir'], dataset_names, full_eval_str)
    print(f"Writing train/valid/test splits to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
     
    filename = os.path.join(out_dir, 'train.tfrecord.gz')
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)
    
    # Save the preprocessor data
    preprocessor.to_json(os.path.join(out_dir, 'preprocessor.json'))

    # Process the validation data
    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = os.path.join(out_dir, 'valid.tfrecord.gz')
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    
    # Save train, valid, and test datasets
    for split, split_name in [(train, 'train'),
                              (valid, 'valid'),
                              (test, 'test')]:
        split[['comp_type','composition', 'id', 'energyperatom']].to_csv(
            os.path.join(out_dir, f'{split_name}.csv.gz'), compression='gzip', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', )
    parser.add_argument('--config-file', type=str, default="config/config.yaml",
                        help='config file to use when building dataset splits and training the model')
    # parser.add_argument('--hypo-structures', type=str, action='append',
    #                    help='path/to/hypothetical-structures.json.gz. Can specify multiple times')

    args = parser.parse_args()

    config_map = utils.load_config_file(args.config_file)
    main(config_map)

    # backup code:
    """
    # So pymatgen doesn't want to take the ISO-8859-1 cifs in the tarball, I have to 
    # re-encode as utf-8 using the following command:
    # for file in *; do iconv -f ISO-8859-1 -t UTF-8 < $file > "../utf8_cifs/$file"; done

    # path to ICSD cifs files 
    cif_icsd_filepath = lambda x: os.path.join(base_dir, 'icsd/structures/{}.cif'.format(x))

    # path to relaxed hypothetical structures
    relax_hypo_filepath = lambda x: os.path.join(base_dir, f'{hypo_type}_hypotheticals',
                                                 'relaxed_original/POSCAR_{}'.format(x))

    # path to unrelaxed hypothetical structures
    unrelax_hypo_filepath = lambda x: os.path.join(base_dir, f'{hypo_type}_hypotheticals',
                                                   'unrelaxed_original/POSCAR_{}'.format(x))

    # check if structures corresponding to all 'id' exists
    # icsd_exists = lambda x: os.path.exists(cif_icsd_filepath(x))
    # icsd['cif_exists'] = icsd.id.apply(icsd_exists)
    # icsd = icsd[icsd.cif_exists]

    # zintl_exists = lambda x: os.path.exists(relax_hypo_filepath(x))
    # hypo_df['cif_exists'] = hypo_df.id.apply(zintl_exists)
    # hypo_df = hypo_df[hypo_df.cif_exists]    

    # Drop ICSDs with fractional occupation
    # to_drop = pd.read_csv('icsd_fracoccupation.csv', header=None)[0]\
    # .str.extract('_(\d{6}).cif').astype(int)[0]
    # data = data[~data.icsdnum.isin(to_drop)]
"""