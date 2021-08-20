import os
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
import pickle

import nfp

from nfp_extensions import CifPreprocessor
tqdm.pandas()

      
if __name__ == '__main__':
    base_dir = "/projects/rlmolecule/shubham/file_transfer/decorations/relaxed"
    # type of hypothetical structures we're working with
    hypo_type = "zintl"
    dataset_name = f"icsd_{hypo_type}" 
    icsd_structures_file = f"{base_dir}/icsd/icsd_energies.csv"
    hypothetical_structures_file = f"{base_dir}/{hypo_type}_hypotheticals/relaxed_energies.csv"
        
    # Read energy data
    icsd_df = pd.read_csv(icsd_structures_file)
    hypo_df = pd.read_csv(hypothetical_structures_file)
    
    # So pymatgen doesn't want to take the ISO-8859-1 cifs in the tarball, I have to 
    # re-encode as utf-8 using the following command:
    # for file in *; do iconv -f ISO-8859-1 -t UTF-8 < $file > "../utf8_cifs/$file"; done

    # path to ICSD cifs files 
    cif_icsd_filepath = lambda x: os.path.join(base_dir, 'icsd/structures/{}.cif'.format(x))

    # path to relaxed hypothetical structures
    relax_hypo_filepath = lambda x: os.path.join(base_dir, f'{hypo_type}_hypotheticals', 'relaxed_original/POSCAR_{}'.format(x))
 
    # path to unrelaxed hypothetical structures
    unrelax_hypo_filepath = lambda x: os.path.join(base_dir, f'{hypo_type}_hypotheticals', 'unrelaxed_original/POSCAR_{}'.format(x))

    
    # check if structures corresponding to all 'id' exists
    #icsd_exists = lambda x: os.path.exists(cif_icsd_filepath(x))
    #icsd['cif_exists'] = icsd.id.apply(icsd_exists)
    #icsd = icsd[icsd.cif_exists]
         
    #zintl_exists = lambda x: os.path.exists(relax_hypo_filepath(x))
    #hypo_df['cif_exists'] = hypo_df.id.apply(zintl_exists)
    #hypo_df = hypo_df[hypo_df.cif_exists]    
    
    # Drop ICSDs with fractional occupation
    #to_drop = pd.read_csv('icsd_fracoccupation.csv', header=None)[0]\
        #.str.extract('_(\d{6}).cif').astype(int)[0]
    #data = data[~data.icsdnum.isin(to_drop)]

    # Try to parse ICSD crystals with pymatgen
    def get_icsd_strc(strc_id):
        try:
            struc = Structure.from_file(cif_icsd_filepath(strc_id), primitive=True)
            #struc.lattice = struc.lattice.cubic(1.0)                  # cubic lattice with a=b=c=1
            #struc.lattice = struc.lattice.tetragonal(1.0,3.0)         # tetragonal lattice with a=b=1,c=3
            #struc.lattice = struc.lattice.orthorhombic(1.0,2.0,3.0)   # orthorhombic lattice with a=1,b=2,c=3
            return struc
        except Exception:
            return None


    # Try to parse pg crystals with pymatgen
    def get_hypo_strc(strc_id):
        try:
            struc = Structure.from_file(relax_hypo_filepath(strc_id), primitive=True)
            #struc.lattice = struc.lattice.cubic(1.0)                  # cubic lattice with a=b=c=1
            #struc.lattice = struc.lattice.tetragonal(1.0,3.0)         # tetragonal lattice with a=b=1,c=3
            #struc.lattice = struc.lattice.orthorhombic(1.0,2.0,3.0)   # orthorhombic lattice with a=1,b=2,c=3
            return struc
        except Exception:
            return None

    
    # record ICSD structures as a column
    print(f"Reading ICSD structures from {cif_icsd_filepath('*')}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        icsd_df['crystal'] = icsd_df.id.progress_apply(get_icsd_strc)


    # record hypothetical structures as a column
    print(f"Reading hypothetical {hypo_type} structures from {relax_hypo_filepath('*')}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hypo_df['crystal'] = hypo_df.id.progress_apply(get_hypo_strc)

    # record parse issues
    icsd_df[icsd_df.crystal.isna()]['id'].to_csv('problems_icsd.csv')
        
    # record parse issues
    hypo_df[hypo_df.crystal.isna()]['id'].to_csv('problems_hypo.csv')


    icsd_df = icsd_df.dropna(subset=['crystal'])
    hypo_df = hypo_df.dropna(subset=['crystal'])
    print(f'{len(icsd_df),len(hypo_df)} icsd, hypothetical crystals after down-selection')
  
    
    # Split the icsd data into training and test sets
    train_icsd, test_icsd  = train_test_split(icsd_df.id.unique(), test_size=500, random_state=1)
    train_icsd, valid_icsd = train_test_split(train_icsd, test_size=500, random_state=1)


    # Split the hypothetical data into training and test sets, such that test/valid sets have one composition per comp_type
    valid_hypo = hypo_df.groupby("comp_type").sample(n=1, random_state=1)
    train_hypo = hypo_df[~hypo_df.composition.isin(valid_hypo.composition)]
    test_hypo  = train_hypo.groupby("comp_type").sample(n=1, random_state=1)
    train_hypo = train_hypo[~train_hypo.composition.isin(test_hypo.composition)]


    # merge train, valid, test sets from icsd and hypo data above
    icsd_train = icsd_df[icsd_df.id.isin(train_icsd)]
    train      = icsd_train.append(train_hypo)

    icsd_valid = icsd_df[icsd_df.id.isin(valid_icsd)]
    hypo_valid = hypo_df[hypo_df.composition.isin(valid_hypo.composition)]
    valid      = icsd_valid.append(hypo_valid)

    icsd_test  = icsd_df[icsd_df.id.isin(test_icsd)]
    hypo_test  = hypo_df[hypo_df.composition.isin(test_hypo.composition)]
    test       = icsd_test.append(hypo_test)

         
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

    out_dir = f'tfrecords_{dataset_name}'
    os.mkdir(out_dir)
     
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
    train[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        os.path.join(out_dir, 'train.csv.gz'), compression='gzip', index=False)
    valid[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        os.path.join(out_dir, 'valid.csv.gz'), compression='gzip', index=False)
    test[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        os.path.join(out_dir, 'test.csv.gz'), compression='gzip', index=False)
