# This file is modified from jongwook/crepe
# For the original, please visit https://github.com/jongwook/crepe/scripts/verify_dataset.py

import os
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire", "tfrecord")
records = []
for grade_name in sorted(os.listdir(dataset_folder)):
    grade_folder = os.path.join(dataset_folder, grade_name)
    for file in sorted(os.listdir(grade_folder)):
        if file.endswith('.tfrecord'):
            records.append(os.path.join(grade_folder, file))

print(len(records), 'records found')

pitches = []
energies = []

for record in tqdm(records):
    for record in tf.python_io.tf_record_iterator(record, options=options):
        example = tf.train.Example()
        example.ParseFromString(record)

        audio = example.features.feature['audio'].float_list.value
        pitch = example.features.feature['pitch'].float_list.value[0]

        try:
            assert len(audio) == 1024
        except AssertionError:
            print("the audio segment length should be 1024 in the dataset!!")

        energies.append(np.linalg.norm(audio))
        pitches.append(pitch)

num_examples = len(pitches)
print(num_examples, "examples found")

energy_bins = np.linspace(0, 20, 21)
energy_hist, _ = np.histogram(energies, bins=energy_bins)
energy_dist = energy_hist / num_examples

print("Energy distribution:")
for e, p in zip(energy_bins, energy_dist):
    print("%.2f" % e, "*" * int(p * 100))

pitch_bins = np.linspace(0, 1000, 21)
pitch_hist, _ = np.histogram(pitches, bins=pitch_bins)
pitch_dist = pitch_hist / num_examples

print("Pitch distribution:")
for f, p in zip(pitch_bins, pitch_dist):
    print("%4d" % f, "*" * int(p * 100))
