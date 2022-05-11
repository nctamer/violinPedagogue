import pyrubberband
import os
import glob
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
from mir_eval.melody import hz2cents
from scipy import interpolate
import matplotlib.pyplot as plt

def accuracies(true_cents, predicted_cents, cent_tolerance=50, return_mean_std_abs_error=False):
    from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    if return_mean_std_abs_error:
        abs_error = np.abs(true_cents-predicted_cents)
        abs_error = abs_error[abs_error<50]   #to get rid of outliers
        return rpa, rca, np.mean(abs_error), np.std(abs_error)
    else:
        return rpa, rca

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


def evaluate_per_equal_temperament_deviation(predicted_file_list, ground_truth_file_list,
                                             pitch_range=None, format='URMP', step=10):
    cents_predicted, cents_ground = [], []
    for i, predicted_i in enumerate(predicted_file_list):
        ground_i = ground_truth_file_list[i]
        if format=='URMP':
            ground = pd.read_csv(ground_i, sep="\t", header=None, names=["time", "frequency"])
        if format=='RESYN':
            ground = pd.read_csv(ground_i, sep=",", header=None, names=["time", "frequency"])

        predicted = pd.read_csv(predicted_i)

        f = interpolate.interp1d(predicted["time"], predicted["frequency"],
                                 kind="cubic", fill_value="extrapolate")
        f0_ground = ground["frequency"].values
        f0_predicted = f(ground["time"])[f0_ground > 0]
        f0_ground = f0_ground[f0_ground > 0]
        if pitch_range:
            range_bool = np.logical_and(f0_ground > pitch_range[0], f0_ground <= pitch_range[1])
            f0_predicted = f0_predicted[range_bool]
            f0_ground = f0_ground[range_bool]
        cents_predicted.append(hz2cents(f0_predicted))
        cents_ground.append(hz2cents(f0_ground))
    cents_ground = np.hstack(cents_ground)
    cents_predicted = np.hstack(cents_predicted)
    ground_eq_deviation = cents_ground - hz2cents(np.array([440]))[0] % 100
    ground_eq_deviation = np.mod(ground_eq_deviation, 100)
    acc = {}
    assert 100%step == 0
    for subregion in range(int(np.ceil((100//step)))):
        subregion = (step/2)*(subregion+1)
        subregion = (subregion, 100-subregion)
        relevant_region = np.logical_or(ground_eq_deviation<subregion[0], ground_eq_deviation>subregion[1])
        rpa5, rca5 = accuracies(cents_ground[relevant_region], cents_predicted[relevant_region], cent_tolerance=5)
        n_samples = sum(relevant_region)
        print(rpa5, rca5, subregion, n_samples)
        acc[subregion] = rpa5
        cents_ground = cents_ground[~relevant_region]
        cents_predicted = cents_predicted[~relevant_region]
        ground_eq_deviation = ground_eq_deviation[~relevant_region]

    return acc


def bach10_evaluate_model_vs_equal_temperament(model_name, pitch_range=None, step=10, instrument=None,
                                               bach10_path=os.path.join(os.path.expanduser("~"), "violindataset",
                                                                        "Bach10-mf0-synth_*c_shifted")):
    search_string = '.RESYN'
    if instrument:
        search_string = instrument + search_string
    dataset_folder = os.path.join(bach10_path, "annotation_stems")
    pitch_tracks_folder = os.path.join(bach10_path, 'pitch_tracks', model_name)
    predicted_file_list = sorted(glob.glob(os.path.join(pitch_tracks_folder, "*" + search_string + ".f0.csv")))
    ground_file_list = sorted(glob.glob(os.path.join(dataset_folder, "*" + search_string + ".csv")))
    assert len(predicted_file_list) == len(
        ground_file_list)  # to ensure we have pitch tracks for all the instrument data
    return evaluate_per_equal_temperament_deviation(predicted_file_list=predicted_file_list,
                                                    ground_truth_file_list=ground_file_list, format='RESYN',
                                                    pitch_range=pitch_range, step=step)

def urmp_evaluate_model_vs_equal_temperament(model_name, pitch_range=None, step=10, instrument=None,
                                  urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP")):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    if not instrument:
        instrument = '*'  # all instruments
    ground_file_list = sorted(glob.glob(os.path.join(dataset_folder, "*/F0s*_" + instrument + "_*.txt")))
    pitch_tracks_folder = os.path.join(urmp_path, 'pitch_tracks', model_name, instrument)
    predicted_file_list = glob.glob(os.path.join(pitch_tracks_folder,
                                                        "*/AuSep*_" + instrument + "_*.f0.csv"))
    predicted_file_list.sort(key = lambda x: '/'.join(x.rsplit('/',2)[1:]))
    assert len(predicted_file_list) == len(ground_file_list)
    return evaluate_per_equal_temperament_deviation(predicted_file_list=predicted_file_list,
                                                    ground_truth_file_list=ground_file_list, format='URMP',
                                                    pitch_range=pitch_range, step=step)


if __name__ == '__main__':
    model_names = {'original': 'original',
                   'violinPedagogue': 'no_pretrain_instrument_model_50_005',
                   'finetuned': 'finetuned_instrument_model_50_005',
                   'violinPedagogueNOshift': 'no_pretrain_standard_no_pitch_shift',
                   'finetunedNOshift': 'finetuned_standard_no_pitch_shift'}

    df_bach = {}
    for show_name, name in model_names.items():
        df_bach[show_name] = bach10_evaluate_model_vs_equal_temperament(name, pitch_range=(190, 4000), step=4,
                                                                        bach10_path=os.path.join(
                                                                            os.path.expanduser("~"), "violindataset",
                                                                            "Bach10-mf0-synth"))
    df_bach = pd.DataFrame(data=df_bach)

    df_vn_bach = {}
    for show_name, name in model_names.items():
        df_vn_bach[show_name] = bach10_evaluate_model_vs_equal_temperament(name, instrument='violin', step=4,
                                                                           bach10_path=os.path.join(
                                                                               os.path.expanduser("~"), "violindataset",
                                                                               "Bach10-mf0-synth"))
    df_vn_bach = pd.DataFrame(data=df_vn_bach)


    df_urmp = {}
    for show_name, name in model_names.items():
        df_urmp[show_name] = urmp_evaluate_model_vs_equal_temperament(name, pitch_range=(190, 4000), step=4)
    df_urmp = pd.DataFrame(data = df_urmp)

    df_vn_urmp = {}
    for show_name, name in model_names.items():
        df_vn_urmp[show_name] = urmp_evaluate_model_vs_equal_temperament(name, instrument='vn', step=4)
    df_vn_urmp = pd.DataFrame(data=df_vn_urmp)




    df_shift = {}
    for show_name, name in model_names.items():
        df_shift[show_name] = bach10_evaluate_model_vs_equal_temperament(name, pitch_range=(190, 4000), step=4)
    df_shift = pd.DataFrame(data = df_shift)

    df_vn_shift = {}
    for show_name, name in model_names.items():
        df_vn_shift[show_name] = bach10_evaluate_model_vs_equal_temperament(name, instrument='violin', step=4)
    df_vn_shift = pd.DataFrame(data = df_vn_shift)

    marker = [',', 'x', '.', 'P', '^']
    cm = 2.2 / 2.54  # centimeters in inches
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(17.2*cm, 7*cm))

    df_vn_urmp.index = [str(_) + '-' + str(_ + 2) for _ in range(0, 50, 2)]
    df_vn_urmp.plot(ax = axes[0,0], grid=True, legend=None, ylabel='RPA5 on violin tracks', title='URMP')

    df_urmp.index = [str(_) + '-' + str(_+2) for _ in range(0, 50, 2)]
    df_urmp.plot(ax = axes[1,0], grid=True, legend=None, ylabel='RPA5 all instruments (vn range)', xlabel='Ground Eq. Temp. Deviation (in cents)')

    df_vn_bach.index = [str(_) + '-' + str(_+2) for _ in range(0, 50, 2)]
    df_vn_bach.plot(ax = axes[0,1], grid=True, legend=None, title='Bach10-mf0-synth')

    df_bach.index = [str(_) + '-' + str(_ + 2) for _ in range(0, 50, 2)]
    df_bach.plot(ax = axes[1,1], grid=True, legend=None, xlabel='Ground Eq. Temp. Deviation (in cents)')

    df_vn_shift.index = [str(_) + '-' + str(_+2) for _ in range(0, 50, 2)]
    df_vn_shift.plot(ax = axes[0,2], grid=True, legend=None, title='Bach10 w/ pitch shifts')

    df_shift.index = [str(_) + '-' + str(_+2) for _ in range(0, 50, 2)]
    df_shift.plot(ax = axes[1,2], grid=True, xlabel='Ground Eq. Temp. Deviation (in cents)', legend=None)
    for axe in axes:
        for ax in axe:
            for i, line in enumerate(ax.get_lines()):
                line.set_marker(marker[i])
    axes[1,2].legend(loc='lower right')
    fig.tight_layout()
    fig.show()
    fig.savefig("fig_pitch_shift.pdf", bbox_inches='tight')
    #for shift in range(0, 101, 10):
    #    bach10_pitch_shift(pitch_shift_cents=shift)

