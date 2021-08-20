import os
import pickle
import math
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import nfp
from nfp_extensions import RBFExpansion

# Initialize the preprocessor class.
from nfp_extensions import CifPreprocessor
preprocessor = CifPreprocessor(num_neighbors=12)

dataset_name = "icsd_zintl"
tfrecords_dir = f'tfrecords_{dataset_name}'

preprocessor.from_json(os.path.join(tfrecords_dir, 'preprocessor.json'))

# Build the tf.data input pipeline
def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        **preprocessor.tfrecord_features,
        **{'energyperatom': tf.io.FixedLenFeature([], dtype=tf.float32)}})

    # All of the array preprocessor features are serialized integer arrays
    for key, val in preprocessor.tfrecord_features.items():
        if val.dtype == tf.string:
            parsed[key] = tf.io.parse_tensor(
                parsed[key], out_type=preprocessor.output_types[key])
    
    # Pop out the prediction target from the stored dictionary as a seperate input
    energyperatom = parsed.pop('energyperatom')
    
    return parsed, energyperatom

# Here, we have to add the prediction target padding onto the input padding
batch_size = 64   # typical batch sizes= 32, 64, 128, 256 (usually batch size=32 or 64 is optimum) 
max_sites = 256
max_bonds = 2048
padded_shapes = (preprocessor.padded_shapes(max_sites=max_sites, max_bonds=max_bonds), [])
padding_values = (preprocessor.padding_values, tf.constant(np.nan, dtype=tf.float32))

train_dataset_file = os.path.join(tfrecords_dir, 'train.tfrecord.gz')
print(f"loading train_dataset_file: {train_dataset_file}")
train_dataset = tf.data.TFRecordDataset(train_dataset_file, compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache()\
    .shuffle(buffer_size=25633)\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset_file = os.path.join(tfrecords_dir, 'valid.tfrecord.gz')
print(f"loading valid_dataset_file: {valid_dataset_file}")
valid_dataset = tf.data.TFRecordDataset(valid_dataset_file, compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache()\
    .shuffle(buffer_size=1000)\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


## Calculate a good initial guess for the site-type embedding layer 
## (the final site-wise contribution to total energy)
# Grab the entire dataset as numpy arrays
site_types, energies = zip(*((inputs['site_features'], outputs) for inputs, outputs
                             in train_dataset.as_numpy_iterator()))
site_types = np.concatenate(site_types)
mean_energies = np.concatenate(energies)

def bincount2D_vectorized(a):
    """https://stackoverflow.com/a/46256361"""
    N = preprocessor.site_classes
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

# This just converts to a count of each element by crystal
site_counts = bincount2D_vectorized(site_types)[:, 2:]

# Linear regression assumes a sum, while we average over sites in the neural network.
# Here, we make the regression target the total energy, not the site-averaged energy
num_sites = site_counts.sum(1)
total_energies = mean_energies * num_sites

# Do the least-squares regression, and stack on zeros for the mask and unknown tokens
output_bias = np.linalg.lstsq(site_counts, total_energies, rcond=None)[0]
output_bias = np.hstack([np.zeros(2), output_bias])


## Build the keras model
site_class = layers.Input(shape=[max_sites], dtype=tf.int64, name='site_features')
distances = layers.Input(shape=[max_bonds], dtype=tf.float32, name='distance')
connectivity = layers.Input(shape=[max_bonds, 2], dtype=tf.int64, name='connectivity')

input_tensors = [site_class, distances, connectivity]

embed_dimension = 256   # typical embedding dimensions= 32, 64, 128, 256
num_messages = 6        # typical number of message passing blocks= 3-6

atom_state = layers.Embedding(preprocessor.site_classes, embed_dimension,
                              name='site_embedding', mask_zero=True)(site_class)

atom_mean = layers.Embedding(preprocessor.site_classes, 1,
                             name='site_mean', mask_zero=True, 
                             embeddings_initializer=tf.keras.initializers.Constant(output_bias))(site_class)

rbf_distance = RBFExpansion(dimension=10,
                            init_max_distance=7,
                            init_gap=30,
                            trainable=True)(distances)

bond_state = layers.Dense(embed_dimension)(rbf_distance)

def message_block(original_atom_state, original_bond_state, connectivity, i):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    
    source_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 1])(connectivity)])
    target_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 0])(connectivity)])    

    # Edge update network
    new_bond_state = layers.Concatenate(name='concat_{}'.format(i))(
        [source_atom, target_atom, bond_state])
    new_bond_state = layers.Dense(
        2*embed_dimension, activation='relu')(new_bond_state)
    new_bond_state = layers.Dense(embed_dimension)(new_bond_state)

    bond_state = layers.Add()([original_bond_state, new_bond_state])

    # message function
    source_atom = layers.Dense(embed_dimension)(source_atom)    
    messages = layers.Multiply()([source_atom, bond_state])
    messages = nfp.Reduce(reduction='sum')(
        [messages, nfp.Slice(np.s_[:, :, 0])(connectivity), atom_state])

    # state transition function
    messages = layers.Dense(embed_dimension, activation='relu')(messages)
    messages = layers.Dense(embed_dimension)(messages)

    atom_state = layers.Add()([original_atom_state, messages])
    
    return atom_state, bond_state

for i in range(num_messages):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity, i)

#atom_state = layers.Dropout(0.25)(atom_state)
atom_state = layers.Dense(1)(atom_state)
atom_state = layers.Add()([atom_state, atom_mean])

out = tf.keras.layers.GlobalAveragePooling1D()(atom_state)

model = tf.keras.Model(input_tensors, [out])

STEPS_PER_EPOCH = math.ceil(25633 / batch_size)  # number of training examples
lr = 1E-4       # too high learning rates can lead to NaN losses 
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr,
  decay_steps=STEPS_PER_EPOCH*50,
  decay_rate=1,
  staircase=False)

wd_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(1E-5,
  decay_steps=STEPS_PER_EPOCH*50,
  decay_rate=1,
  staircase=False)

optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule, global_clipnorm=1.)

model.compile(loss='mae', optimizer=optimizer)

model_name = f'trained_model_{dataset_name}'

if not os.path.exists(model_name):
    os.makedirs(model_name)

# Make a backup of the job submission script
shutil.copy(__file__, model_name)

filepath = model_name + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')


if __name__ == "__main__":
    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=100,
              callbacks=[checkpoint, csv_logger],
              verbose=1)
