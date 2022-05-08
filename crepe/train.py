import sys
from config import *
from datasets import *
from evaluation import accuracies
from tensorflow import keras
import os
import itertools
import numpy as np
from predict import bach10_extract_pitch_with_model, urmp_all_instruments_extract_pitch_with_model, extract_pitch_with_model


validation_set_names = ["monoSuzuki", "monoDancla", "monoWohlfahrt", "monoSitt",
                        "monoKayser", "monoMazas", "monoDontOp37", "monoKreutzer",
                        "monoFiorillo", "monoRode", "monoGavinies", "monoDontOp35"]


def prepare_datasets(parent_folder, methods):

    train, validation = [], []
    for method in methods:
        train_per_method, validation_per_method = [], []
        paths = sorted(os.listdir(os.path.join(parent_folder, method)))
        split_paths = [_.split('_', 3) for _ in paths]
        for split_path in split_paths:
            if int(split_path[2]) % 5 == 3:
                if ".shiftedRESYN." not in split_path[3]:
                    validation_per_method.append(os.path.join(parent_folder, method, '_'.join(split_path)))
            else:
                train_per_method.append(os.path.join(parent_folder, method, '_'.join(split_path)))

        train.append(train_per_method)
        if validation_per_method:
            validation.append(validation_per_method)

    train = train_dataset(*train, batch_size=options['batch_size'], augment=options['augment'])
    print("Train dataset:", train, file=sys.stderr)

    v = []
    for name in validation:
        print("Collecting validation set {}:".format(name), file=sys.stderr)
        dataset = validation_dataset(name, seed=42, take=100).take(options['validation_take']).collect(verbose=True)
        v.append(dataset)

    return train, v


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_sets, local_average=False):
        super().__init__()
        self.val_sets = [(audio, to_weighted_average_cents(pitch)) for audio, pitch in val_sets]
        self.local_average = local_average
        self.to_cents = local_average and to_local_average_cents or to_weighted_average_cents
        self.prefix = local_average and 'local-average-' or 'default-'
        self.mae = np.inf
        for filename in ["mae.tsv", "rpa.tsv", "rca.tsv", "rpa5.tsv", "rca5.tsv"]:
            with open(log_path(self.prefix + filename), "w") as f:
                f.write('\t'.join(validation_set_names) + '\n')

    def on_epoch_end(self, epoch, logs={}):
        names = list(validation_set_names)
        print(file=sys.stderr)

        MAEs = []
        RPAs = []
        RCAs = []
        RPA5 = []
        RCA5 = []

        print("Epoch {}, validation accuracies (local_average = {})".format(epoch + 1, self.local_average), file=sys.stderr)
        for audio, true_cents in self.val_sets:
            predicted = self.model.predict(audio)
            predicted_cents = self.to_cents(predicted)
            diff = np.abs(true_cents - predicted_cents)
            mae = np.mean(diff[np.isfinite(diff)])
            rpa, rca = accuracies(true_cents, predicted_cents, cent_tolerance=10)
            rpa5, rca5 = accuracies(true_cents, predicted_cents, cent_tolerance=5)
            nans = np.mean(np.isnan(diff))
            print("{}: MAE = {}, RPA = {}, RCA = {}, nans = {}".format(names.pop(0), mae, rpa, rca, nans), file=sys.stderr)
            MAEs.append(mae)
            RPAs.append(rpa)
            RCAs.append(rca)
            RPA5.append(rpa5)
            RCA5.append(rca5)
        self.mae = np.mean(MAEs)
        logs["mae"] = self.mae

        with open(log_path(self.prefix + "mae.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % mae for mae in MAEs]) + '\n')
        with open(log_path(self.prefix + "rpa.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in RPAs]) + '\n')
        with open(log_path(self.prefix + "rca.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rca for rca in RCAs]) + '\n')
        with open(log_path(self.prefix + "rpa5.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in RPA5]) + '\n')
        with open(log_path(self.prefix + "rca5.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rca for rca in RCA5]) + '\n')

        print(file=sys.stderr)


def main():
    model_name = 'violin_range'
    tfrecord_folder = 'tfrecord_standard_iter2_finetuned_standard'
    options["load_model_weights"] = "models/original.h5"

    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "monophonic_etudes", tfrecord_folder)
    names = sorted([_ for _ in os.listdir(dataset_folder) if (_.startswith('L') or _.startswith('mono'))])
    train_set, val_sets = prepare_datasets(dataset_folder, names)
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()


    options["save_model_weights"] = model_name + ".h5"
    options["steps_per_epoch"] = 1000
    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_set), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [
                            PitchAccuracyCallback(val_sets, local_average=True)
                        ] + [
                            keras.callbacks.EarlyStopping(monitor='mae', mode="min", patience=100, min_delta=0.01, verbose=1, restore_best_weights=True)
                        ],
                        validation_data=val_data)


    # Bach10-mf0-synth
    bach10_extract_pitch_with_model(model_name, model, viterbi=False, verbose=1)
    for pitch_shift in range(0, 101, 10):
        bachpath = os.path.join(os.path.expanduser("~"), "violindataset", "Bach10-mf0-synth") \
                   + '_' + str(pitch_shift) + 'c_shifted'
        bach10_extract_pitch_with_model(options["save_model_weights"], model,
                                        bach10_path=bachpath, viterbi=False, verbose=1)

    # ViolinPedagogue
    extract_pitch_with_model(model_name=model_name, model=model,
                             main_dataset_folder=os.path.join(os.path.expanduser("~"),
                                                              "violindataset", "monophonic_etudes"),
                             save_activation=False, viterbi=True, verbose=0)

    # URMP
    urmp_all_instruments_extract_pitch_with_model(model_name, model, viterbi=False, verbose=1)




if __name__ == "__main__":
    main()
