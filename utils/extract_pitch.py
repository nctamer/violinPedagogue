import librosa
import os
import glob
import numpy as np
import pandas as pd
import json
from scipy import interpolate
from mir_eval.melody import hz2cents

try:
    import marl_crepe as mycrepe
except ModuleNotFoundError:
    import sys

    # Add the ptdraft folder path to the sys.path list
    sys.path.append('..')
    import marl_crepe as mycrepe


def accuracies(true_cents, predicted_cents, cent_tolerance=50):
    from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    return rpa, rca


def predict_from_file_list(audio_files, output_f0_files, model_path, activation_files=None, viterbi=True, verbose=1):
    for index, audio_file in enumerate(audio_files):
        output_f0_file = output_f0_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        time, frequency, confidence, activation = mycrepe.predict(audio, sr, model_path,
                                                                  viterbi=viterbi, combined_viterbi=False,
                                                                  verbose=verbose)
        df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence},
                          columns=["time", "frequency", "confidence"])
        df.to_csv(output_f0_file, index=False)
        if activation_files:
            output_activation_file = activation_files[index]
            np.save(output_activation_file, activation)
    return


def urmp_extract_pitch_with_model(model_name, instrument='vn',
                                  urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP"),
                                  viterbi=False, verbose=1):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    out_folder = os.path.join(urmp_path, 'pitch_tracks', model_name)

    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')

    audio_files, output_f0_files = [], []
    for track in sorted(os.listdir(dataset_folder)):
        if track[0].isdigit():
            stems = sorted(glob.glob(os.path.join(dataset_folder, track, "AuSep*" + instrument + "*.wav")))
            if len(stems) > 0:
                if not os.path.exists(os.path.join(out_folder, track)):
                    # Create a new directory because it does not exist
                    os.makedirs(os.path.join(out_folder, track))
                new_audio_files = stems
                audio_files.extend(new_audio_files)
                output_f0_files.extend(
                    [os.path.join(out_folder, track, os.path.basename(_)[:-3] + "f0.csv") for _ in new_audio_files])

    predict_from_file_list(audio_files, output_f0_files, model_path, viterbi=viterbi, verbose=verbose)
    return


def extract_pitch_with_model(model_name, main_dataset_folder=os.path.join(os.path.expanduser("~"), "violindataset",
                                                                          "graded_repertoire"),
                             viterbi=True, save_activation=False, verbose=1):
    OUT_FOLDER = os.path.join(main_dataset_folder, 'pitch_tracks', model_name)
    AUDIO_FORMAT = ".mp3"
    GRADES = sorted(os.listdir(main_dataset_folder))

    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')

    audio_files, output_f0_files, activation_files = [], [], []
    activation_folder = os.path.join(main_dataset_folder, 'activations', model_name)
    for grade in sorted(GRADES)[::-1]:
        if os.path.isdir(os.path.join(main_dataset_folder, grade)):
            if not os.path.exists(os.path.join(OUT_FOLDER, grade)):
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(OUT_FOLDER, grade))
        new_audio_files = sorted(glob.glob(os.path.join(main_dataset_folder, grade, "*" + AUDIO_FORMAT)))
        audio_files.extend(new_audio_files)
        output_f0_files.extend(
            [os.path.join(OUT_FOLDER, grade, os.path.basename(_)[:-3] + "f0.csv") for _ in new_audio_files])
        if save_activation:
            if not os.path.exists(os.path.join(activation_folder, grade)):
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(activation_folder, grade))
                activation_files.extend(
                    [os.path.join(activation_folder, grade, os.path.basename(_)[:-3] + "npy") for _ in new_audio_files])

    predict_from_file_list(audio_files, output_f0_files, model_path,
                           activation_files=activation_files, viterbi=viterbi, verbose=verbose)
    return


def single_file_extract_pitch_with_model(audio_file, output_f0_file=None, model_name='original',
                                         viterbi=True, verbose=1):
    if not output_f0_file:
        output_f0_file = audio_file[:-3] + 'f0.csv'
    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')
    predict_from_file_list(audio_files=[audio_file], output_f0_files=[output_f0_file], model_path=model_path,
                           viterbi=viterbi, verbose=verbose)
    return


def evaluate(predicted_file_list, ground_truth_file_list):
    cents_predicted, cents_ground = [], []
    for i, predicted_i in enumerate(predicted_file_list):
        ground_i = ground_truth_file_list[i]
        ground = pd.read_csv(ground_i, sep="\t", header=None, names=["time", "frequency"])

        predicted = pd.read_csv(predicted_i)

        f = interpolate.interp1d(predicted["time"], predicted["frequency"],
                                 kind="cubic", fill_value="extrapolate")
        f0_ground = ground["frequency"].values
        f0_predicted = f(ground["time"])[f0_ground > 0]
        f0_ground = f0_ground[f0_ground > 0]
        cents_predicted.append(hz2cents(f0_predicted))
        cents_ground.append(hz2cents(f0_ground))
    cents_ground = np.hstack(cents_ground)
    cents_predicted = np.hstack(cents_predicted)
    rpa50, rca50 = accuracies(cents_ground, cents_predicted, cent_tolerance=50)
    rpa25, rca25 = accuracies(cents_ground, cents_predicted, cent_tolerance=25)
    rpa10, rca10 = accuracies(cents_ground, cents_predicted, cent_tolerance=10)
    rpa5, rca5 = accuracies(cents_ground, cents_predicted, cent_tolerance=5)
    return {"rpa50": rpa50, "rpa25": rpa25, "rpa10": rpa10, "rpa5": rpa5,
            "rca50": rca50, "rca25": rca25, "rca10": rca10, "rca5": rca5}


def urmp_evaluate_model(model_name, instrument='vn',
                        urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP")):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    pitch_tracks_folder = os.path.join(urmp_path, 'pitch_tracks', model_name)
    predicted_file_list = sorted(glob.glob(os.path.join(pitch_tracks_folder, "*/AuSep*" + instrument + "*.f0.csv")))
    ground_file_list = sorted(glob.glob(os.path.join(dataset_folder, "*/F0s*" + instrument + "*.txt")))
    assert len(predicted_file_list) == len(
        ground_file_list)  # to ensure we have pitch tracks for all the instrument data
    return evaluate(predicted_file_list=predicted_file_list, ground_truth_file_list=ground_file_list)


def urmp_evaluate_all(instrument="vn", urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP")):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    ground_file_list = sorted(glob.glob(os.path.join(dataset_folder, "*/F0s*" + instrument + "*.txt")))
    evaluation = {}
    for model_name in os.listdir(os.path.join(urmp_path, 'pitch_tracks')):
        pitch_tracks_folder = os.path.join(urmp_path, 'pitch_tracks', model_name)
        if os.path.isdir(pitch_tracks_folder):
            predicted_file_list = sorted(glob.glob(os.path.join(pitch_tracks_folder,
                                                                "*/AuSep*" + instrument + "*.f0.csv")))
            assert len(predicted_file_list) == len(ground_file_list)
            evaluation[model_name] = evaluate(predicted_file_list=predicted_file_list,
                                              ground_truth_file_list=ground_file_list)
            print(model_name)
            eval_string = ""
            for key, value in evaluation[model_name].items():
                eval_string = eval_string + "{:s}: {:.3f}%   ".format(key, 100 * value)
            print(eval_string + "\n")
    json.dump(evaluation, open(os.path.join(urmp_path, "pitch_tracks", "evaluation.json"), "w"))


if __name__ == '__main__':
    new_model_name = 'original'
    #urmp_extract_pitch_with_model(new_model_name, instrument="vn", viterbi=False, verbose=1)
    #urmp_evaluate_all(instrument="vn")
    extract_pitch_with_model(model_name=new_model_name,
                             main_dataset_folder=os.path.join(os.path.expanduser("~"),
                                                              "violindataset", "monophonic_etudes"),
                             save_activation=False, viterbi=True, verbose=1)
    #for new_model_name in new_model_names:
    #    extract_pitch_with_model(model_name=new_model_name, save_activation=True, viterbi=True, verbose=0)


    # os.chdir(os.getcwd()+'/utils')
    # single_file_extract_pitch_with_model('kurdili_solo_violin.wav', model_name='iter1')
    # single_file_extract_pitch_with_model('kurdili_solo_violin.wav', model_name='original', viterbi='weird')

