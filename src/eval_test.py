import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.metrics import mean_squared_error 

import gzip
import json
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
from pymatgen.core.structure import Structure
from tqdm import tqdm, trange
import monotonic
mtime = monotonic.time.time
t0 = mtime()

import nfp
from nfp_extensions import RBFExpansion, CifPreprocessor
import utils


dataset_name = "icsd_zintl"

# Initialize the preprocessor class.
preprocessor = CifPreprocessor(num_neighbors=12)
preprocessor.from_json(f'tfrecords_{dataset_name}/preprocessor.json')
    
model_name = f'trained_model_{dataset_name}'
model = tf.keras.models.load_model(
    f'{model_name}/best_model.hdf5',
    custom_objects={**nfp.custom_objects, **{'RBFExpansion': RBFExpansion}})

test = pd.read_csv(f'tfrecords_{dataset_name}/test.csv.gz')
print(test)
icsd = pd.read_csv('/projects/rlmolecule/shubham/file_transfer/decorations/relaxed/icsd/icsd_energies.csv')
print(icsd.head())
hypo_zintl = pd.read_csv('/projects/rlmolecule/shubham/file_transfer/decorations/relaxed/zintl_hypotheticals/relaxed_energies.csv')
print(hypo_zintl.head())

test_icsd = test[test.id.isin(icsd.id)].reset_index(drop=True)
test_hypo = test[~test.id.isin(icsd.id)].reset_index(drop=True)
print(test_icsd)
print(test_hypo)

# load the icsd and hypothetical datasets
icsd_structures_file = "inputs/icsd_structures.json.gz"
icsd_structures = utils.load_structures_from_json(icsd_structures_file)

hypo_structures_file = "inputs/zintl_relaxed_structures.json.gz"
hypo_structures = utils.load_structures_from_json(hypo_structures_file)


icsd_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(icsd_structures[id], train=False)
             for id in tqdm(test_icsd.id)),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=32,
                  padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                  padding_values=preprocessor.padding_values)


hypo_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(hypo_structures["POSCAR_"+id], train=False)
             for id in tqdm(test_hypo.id)),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=32,
                  padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                  padding_values=preprocessor.padding_values)


predictions_icsd = model.predict(icsd_dataset)
predictions_hypo   = model.predict(hypo_dataset)

test_icsd['predicted_energyperatom'] = predictions_icsd
test_hypo['predicted_energyperatom']   = predictions_hypo

df = pd.DataFrame()
df = df.append(test_icsd)
df = df.append(test_hypo)
df.to_csv('predicted_test.csv', index=False)

##MAE and RMSE##
realVals = df.energyperatom
predictedVals = df.predicted_energyperatom
rmse = mean_squared_error(realVals, predictedVals, squared = False)

f = open('mae_test.txt','w')
print(f'Test MAE: {(df.energyperatom - df.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom', file=f)
print("Test RMSE:", rmse, file=f)
print(f'ICSD MAE: {(test_icsd.energyperatom - test_icsd.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom', file=f)
print(f'Hypo MAE: {(test_hypo.energyperatom - test_hypo.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom', file=f)
f.close()

elapsed = mtime()-t0
f_time = open('time_test.txt', 'w')
print("elaspsed_time:", elapsed, file=f_time)
f_time.close()

