import pyrubberband
import os
import glob
import librosa
import pandas as pd
import numpy as np
import soundfile as sf

def pitch_shift_from_file_list(audio_files, f0_files, pitch_shift_cents, shifted_audio_files, shifted_f0_files):
    pitch_shift_semitones = pitch_shift_cents/100
    for index, audio_file in enumerate(audio_files):
        f0_file = f0_files[index]
        shifted_f0_file = shifted_f0_files[index]
        shifted_audio_file = shifted_audio_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        shifted_audio = pyrubberband.pitch_shift(audio, sr, n_steps=pitch_shift_semitones)
        f0s = pd.read_csv(f0_file, sep=",", header=None, names=["time", "frequency"])
        f0s['frequency'] = f0s['frequency'] * pow(2, (pitch_shift_cents / 1200))
        f0s.to_csv(shifted_f0_file, sep=",", header=None, index=False)
        sf.write(shifted_audio_file, shifted_audio, sr, 'PCM_24')

    return


def bach10_pitch_shift(bach10_path=os.path.join(os.path.expanduser("~"), "violindataset", "Bach10-mf0-synth"),
                       pitch_shift_cents=0, shifted_bach10_path=None):
    if not shifted_bach10_path:
        shifted_bach10_path = bach10_path + '_' + str(pitch_shift_cents) + 'c_shifted'
    audio_folder = os.path.join(bach10_path, "audio_stems")
    f0_folder = os.path.join(bach10_path, "annotation_stems")
    shifted_audio_folder = os.path.join(shifted_bach10_path, "audio_stems")
    shifted_f0_folder = os.path.join(shifted_bach10_path, "annotation_stems")
    for out_folder in [shifted_audio_folder, shifted_f0_folder]:
        if not os.path.exists(os.path.join(out_folder)):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(out_folder))
    audio_files, f0_files, shifted_audio_files, shifted_f0_files = [], [], [], []
    for track in sorted(os.listdir(audio_folder)):
        if track[0].isdigit():
            audio_files.append(os.path.join(audio_folder, track))
            shifted_audio_files.append(os.path.join(shifted_audio_folder, track))
            f0_files.append(os.path.join(f0_folder, track[:-3] + "csv"))
            shifted_f0_files.append(os.path.join(shifted_f0_folder, track[:-3] + "csv"))
    pitch_shift_from_file_list(audio_files, f0_files, pitch_shift_cents, shifted_audio_files, shifted_f0_files)
    return


def urmp_extract_pitch_with_model(model_name, instrument='vn',
                                  urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP"),
                                  viterbi=False, verbose=1):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    out_folder = os.path.join(urmp_path, 'pitch_tracks', model_name, instrument)

    model_path = os.path.join('..', 'crepe', 'models', model_name + '.h5')

    audio_files, f0_files, shifted_audio_files, shifted_f0_files = [], [], [], []
    for track in sorted(os.listdir(dataset_folder)):
        if track[0].isdigit():
            stems = sorted(glob.glob(os.path.join(dataset_folder, track, "AuSep*_" + instrument + "_*.wav")))
            if len(stems) > 0:
                if not os.path.exists(os.path.join(out_folder, track)):
                    # Create a new directory because it does not exist
                    os.makedirs(os.path.join(out_folder, track))
                new_audio_files = stems
                audio_files.extend(new_audio_files)
                output_f0_files.extend(
                    [os.path.join(out_folder, track, os.path.basename(_)[:-3] + "csv") for _ in new_audio_files])

    pitch_shift_from_file_list(audio_files, output_f0_files, model_path)
    return


if __name__ == '__main__':
    for shift in range(0, 101, 10):
        bach10_pitch_shift(pitch_shift_cents=shift)

