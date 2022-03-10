import librosa
import os
import glob
import pandas as pd
try:
    import marl_crepe as mycrepe
except ModuleNotFoundError:
    import sys
    # Add the ptdraft folder path to the sys.path list
    sys.path.append('..')
    import marl_crepe as mycrepe


def predict_from_file_list(audio_files, output_f0_files, model_path):
    for index, audio_file in enumerate(audio_files):
        output_f0_file = output_f0_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        time, frequency, confidence, _ = mycrepe.predict(audio, sr, model_path, viterbi=True)
        df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence},
                          columns=["time", "frequency", "confidence"])
        df.to_csv(output_f0_file, index=False)
    return


def urmp_extract_pitch_with_model(model_name, instrument='vn',
                                  urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP")):

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

    predict_from_file_list(audio_files, output_f0_files, model_path)
    return


def extract_pitch_with_model(model_name):

    FOLDER = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
    OUT_FOLDER = os.path.join(FOLDER, 'pitch_tracks', model_name)
    AUDIO_FORMAT = ".mp3"
    GRADES = ["L1", "L2", "L3", "L4", "L5", "L6"]

    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')

    audio_files, output_f0_files = [], []
    for grade in GRADES:
        if not os.path.exists(os.path.join(OUT_FOLDER, grade)):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(OUT_FOLDER, grade))
        new_audio_files = sorted(glob.glob(os.path.join(FOLDER, grade, "*" + AUDIO_FORMAT)))
        audio_files.extend(new_audio_files)
        output_f0_files.extend(
            [os.path.join(OUT_FOLDER, grade, os.path.basename(_)[:-3] + "f0.csv") for _ in new_audio_files])

    predict_from_file_list(audio_files, output_f0_files, model_path)
    return


if __name__ == '__main__':
    urmp_extract_pitch_with_model("original", instrument="vn")
    urmp_extract_pitch_with_model("firstRun", instrument="vn")
    #extract_pitch_with_model(model_name='firstRun')
