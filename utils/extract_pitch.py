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

GRADES = ["L1", "L2", "L3", "L4", "L5", "L6"]


def extract_pitch_with_model(model_name):

    FOLDER = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
    OUT_FOLDER = os.path.join(FOLDER, 'pitch_tracks', model_name)
    AUDIO_FORMAT = ".mp3"

    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')

    if not os.path.exists(OUT_FOLDER):
        # Create a new directory because it does not exist
        os.makedirs(OUT_FOLDER)
    audio_files, output_f0_files = [], []
    for grade in GRADES:
        new_audio_files = sorted(glob.glob(os.path.join(FOLDER, grade, "*" + AUDIO_FORMAT)))
        audio_files.extend(new_audio_files)
        output_f0_files.extend(
            [os.path.join(OUT_FOLDER, grade, os.path.basename(_)[:-3] + "f0.csv") for _ in new_audio_files])

    for index, audio_file in enumerate(audio_files):
        output_f0_file = output_f0_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        time, frequency, confidence, _ = mycrepe.predict(audio, sr, model_path, viterbi=True)
        df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence},
                          columns=["time", "frequency", "confidence"])
        df.to_csv(output_f0_file, index=False)
    return


if __name__ == '__main__':
    extract_pitch_with_model(model_name='firstRun')
