import sys
from config import *
from datasets import *
from evaluation import accuracies
from tensorflow import keras
import os
import itertools
import numpy as np

validation_set_names = ["monoKreutzer", "monoKayser", "monoWohlfahrt"]


def prepare_datasets(parent_folder, grades) -> (Dataset, (np.ndarray, np.ndarray)):
    gp = [[(grade, _.split('_')[1]) for _ in os.listdir(os.path.join(parent_folder, grade))] for grade in grades]
    _, player = zip(*set(itertools.chain.from_iterable(gp)))
    p, p_count = zip(*[(pl, player.count(pl)) for pl in sorted(list(set(player)))])
    p, p_count = np.array(p), np.array(p_count)
    possible_validation_players = p[p_count == 1]
    possible_validation_players = ["SunKim", "TimRohwer"]
    train, validation = [], []
    for grade in grades:
        train_per_grade, validation_per_grade = [], []
        paths = os.listdir(os.path.join(parent_folder, grade))
        grade_players = [_.split('_')[1] for _ in paths]
        try:
            val_players_this_grade = list(set(grade_players).intersection(possible_validation_players))
            player_counts = [grade_players.count(player) for player in val_players_this_grade]
            validation_player = val_players_this_grade[np.argmin(player_counts)]
        except:
            print('no validation player for grade', grade)
            validation_player = 'NONE'
        for path in paths:
            if validation_player in path:
                if ".shiftedRESYN." not in path:
                    validation_per_grade.append(os.path.join(parent_folder, grade, path))
            else:
                train_per_grade.append(os.path.join(parent_folder, grade, path))
        train.append(train_per_grade)
        if validation_per_grade:
            validation.append(validation_per_grade)

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
        for filename in ["mae.tsv", "rpa.tsv", "rca.tsv", "rpa5.tsv", "rca5.tsv"]:
            with open(log_path(self.prefix + filename), "w") as f:
                f.write('\t'.join(validation_set_names) + '\n')

    def on_epoch_end(self, epoch, logs=None):
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
    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "monophonic_etudes", "tfrecord")
    names = sorted([_ for _ in os.listdir(dataset_folder) if (_.startswith('L') or _.startswith('mono'))])
    train_set, val_sets = prepare_datasets(dataset_folder, names)
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()

    options["load_model_weights"] = "models/original.h5"
    options["save_model_weights"] = "april.h5"
    options["steps_per_epoch"] = 1000
    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_set), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [
                            PitchAccuracyCallback(val_sets, local_average=True)
                        ],
                        validation_data=val_data)


if __name__ == "__main__":
    main()
