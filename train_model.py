import os
import pickle
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import nfp
from nfp_extensions import RBFExpansion

with open('tfrecords/preprocessor.p', 'rb') as f:
    preprocessor = pickle.load(f)

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
batch_size = 128
max_sites = 256
max_bonds = 2048
padded_shapes = (preprocessor.padded_shapes(max_sites=max_sites, max_bonds=max_bonds), [])
padding_values = (preprocessor.padding_values, tf.constant(np.nan, dtype=tf.float32))

train_dataset = tf.data.TFRecordDataset('tfrecords/train.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache()\
    .shuffle(buffer_size=10000)\
    .repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset('tfrecords/valid.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache()\
    .shuffle(buffer_size=1000)\
    .repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


# Build the keras model
site_class = layers.Input(shape=[max_sites], dtype=tf.int64, name='site_features')
distances = layers.Input(shape=[max_bonds], dtype=tf.float32, name='distance')
connectivity = layers.Input(shape=[max_bonds, 2], dtype=tf.int64, name='connectivity')

input_tensors = [site_class, distances, connectivity]

embed_dimension = 128
num_messages = 4

atom_state = layers.Embedding(preprocessor.site_classes, embed_dimension,
                              name='site_embedding', mask_zero=True)(site_class)

atom_mean = layers.Embedding(preprocessor.site_classes, 1,
                             name='site_mean', mask_zero=True)(site_class)

rbf_distance = RBFExpansion(dimension=embed_dimension,
                            init_max_distance=7,
                            init_gap=30,
                            trainable=False)(distances)
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

atom_state = layers.Dropout(0.25)(atom_state)
atom_state = layers.Dense(1)(atom_state)
atom_state = layers.Add()([atom_state, atom_mean])

out = tf.keras.layers.GlobalAveragePooling1D()(atom_state)

model = tf.keras.Model(input_tensors, [out])

STEPS_PER_EPOCH = math.ceil(12609 / batch_size)  # number of training examples
lr = 3E-4
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr,
  decay_steps=STEPS_PER_EPOCH*50,
  decay_rate=1,
  staircase=False)

model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr_schedule))

model_name = 'trained_model'

if not os.path.exists(model_name):
    os.makedirs(model_name)

filepath = model_name + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=500,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=math.ceil(500/batch_size),
          callbacks=[checkpoint, csv_logger],
          verbose=1)
