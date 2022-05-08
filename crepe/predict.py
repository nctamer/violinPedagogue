import librosa
import os
import glob
import numpy as np
import pandas as pd
import json
from scipy import interpolate
from mir_eval.melody import hz2cents
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
import sys
import re
from scipy.stats import norm
try:
    from datasets import to_local_average_cents
except ImportError:
    from crepe.datasets import to_local_average_cents


URMP_INSTRUMENTS = ["vn", "va", "vc", "db", "fl", "ob", "cl", "sax", "bn", "tpt", "hn", "tbn", "tba"]
# the model is trained on 16kHz audio
model_srate = 16000

def output_path(file, suffix, output_dir):
    """
    return the output path of an output file corresponding to a wav file
    """
    path = re.sub(r"(?i).wav$", suffix, file)
    if output_dir is not None:
        path = os.path.join(output_dir, os.path.basename(path))
    return path


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from librosa.sequence import viterbi_discriminative

    # transition probabilities inducing continuous pitch
    # big changes are penalized with one order of magnitude
    transition = gaussian_filter1d(np.eye(360), 30) + 9*gaussian_filter1d(np.eye(360), 2)
    transition = transition / np.sum(transition, axis=1)[:, None]

    p = salience/salience.sum(axis=1)[:, None]
    p[np.isnan(p.sum(axis=1)), :] = np.ones(360) * 1/360
    path = viterbi_discriminative(p.T, transition)

    return path, np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                           range(len(path))])


def to_weird_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    print('WARNING!! USING THE WEIRD VITERBI ALGORITHM FROM THE ORIGINAL CREPE')
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return path, np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                           range(len(path))])


def predict(audio, sr, model,
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model,
                                center=center, step_size=step_size,
                                verbose=verbose)

    if viterbi == "weird":
        path, cents = to_weird_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    elif viterbi:
        # NEW!! CONFIDENCE IS NO MORE THE MAX ACTIVATION! CORRECTED TO BE CALCULATED ALONG THE PATH!
        path, cents = to_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    else:
        cents = to_local_average_cents(activation)
        confidence = activation.max(axis=1)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation


def get_activation(audio, sr, model, center=True, step_size=10,
                   verbose=1):
    """
    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model : keras model
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)
    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    # run prediction and convert the frequency bin weights to Hz
    return model.predict(frames, verbose=verbose)



def accuracies(true_cents, predicted_cents, cent_tolerance=50):
    from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    return rpa, rca


def predict_from_file_list(audio_files, output_f0_files, model, activation_files=None, viterbi=True, verbose=1):
    for index, audio_file in enumerate(audio_files):
        output_f0_file = output_f0_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        time, frequency, confidence, activation = predict(audio, sr, model, viterbi=viterbi, verbose=verbose)
        df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence},
                          columns=["time", "frequency", "confidence"])
        df.to_csv(output_f0_file, index=False)
        if activation_files:
            output_activation_file = activation_files[index]
            np.save(output_activation_file, activation)
    return


def external_data_extract_pitch_with_model(model_name, model, external_data_path, viterbi=True, verbose=1):
    dataset_folder = os.path.join(external_data_path, "audio")
    out_name = model_name
    if viterbi == 'weird':
        out_name += '_weird'
    elif viterbi == False:
        out_name += '_no_viterbi'
    out_folder = os.path.join(external_data_path, 'pitch_tracks', out_name)
    if not os.path.exists(os.path.join(out_folder)):
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(out_folder))

    audio_files, output_f0_files = [], []
    for track in sorted(os.listdir(dataset_folder)):
        audio_files.append(os.path.join(dataset_folder, track))
        output_f0_files.append(os.path.join(out_folder, track[:-3] + "f0.csv"))
    predict_from_file_list(audio_files, output_f0_files, model, viterbi=viterbi, verbose=verbose)
    return


def bach10_extract_pitch_with_model(model_name, model, bach10_path=os.path.join(os.path.expanduser("~"),
                                                                         "violindataset", "Bach10-mf0-synth"),
                                    viterbi=False, verbose=1):
    dataset_folder = os.path.join(bach10_path, "audio_stems")
    out_folder = os.path.join(bach10_path, 'pitch_tracks', model_name)
    if not os.path.exists(os.path.join(out_folder)):
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(out_folder))

    audio_files, output_f0_files = [], []
    for track in sorted(os.listdir(dataset_folder)):
        if track[0].isdigit():
            audio_files.append(os.path.join(dataset_folder, track))
            output_f0_files.append(os.path.join(out_folder, track[:-3] + "f0.csv"))
    predict_from_file_list(audio_files, output_f0_files, model, viterbi=viterbi, verbose=verbose)
    return


def urmp_extract_pitch_with_model(model_name, model, instrument='vn',
                                  urmp_path=os.path.join(os.path.expanduser("~"), "violindataset", "URMP"),
                                  viterbi=False, verbose=1):
    dataset_folder = os.path.join(urmp_path, "Dataset")
    out_folder = os.path.join(urmp_path, 'pitch_tracks', model_name, instrument)

    audio_files, output_f0_files = [], []
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
                    [os.path.join(out_folder, track, os.path.basename(_)[:-3] + "f0.csv") for _ in new_audio_files])

    predict_from_file_list(audio_files, output_f0_files, model, viterbi=viterbi, verbose=verbose)
    return


def urmp_all_instruments_extract_pitch_with_model(model_name, model, urmp_path=os.path.join(os.path.expanduser("~"),
                                                                                     "violindataset", "URMP"),
                                                  viterbi=False, verbose=1):
    for instrument in URMP_INSTRUMENTS:
        urmp_extract_pitch_with_model(model_name, model, urmp_path=urmp_path,
                                      instrument=instrument, viterbi=viterbi, verbose=verbose)
    return


def extract_pitch_with_model(model_name, model, main_dataset_folder=os.path.join(os.path.expanduser("~"), "violindataset",
                                                                          "graded_repertoire"),
                             viterbi=True, save_activation=False, verbose=1):
    out_name = model_name
    if viterbi == 'weird':
        out_name += '_weird'
    elif viterbi == False:
        out_name += '_no_viterbi'
    OUT_FOLDER = os.path.join(main_dataset_folder, 'pitch_tracks', out_name)
    AUDIO_FORMAT = ".mp3"
    GRADES = sorted([_ for _ in os.listdir(main_dataset_folder) if (_.startswith('L') or _.startswith('mono'))])

    audio_files, output_f0_files, activation_files = [], [], []
    activation_folder = os.path.join(main_dataset_folder, 'activations', out_name)
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

    predict_from_file_list(audio_files, output_f0_files, model,
                           activation_files=activation_files, viterbi=viterbi, verbose=verbose)
    return



if __name__ == '__main__':
    new_model_name = 'finetuned_standard_iter2'

    # Bach10-mf0-synth
    bach10_extract_pitch_with_model(new_model_name, viterbi=False, verbose=1)
    for pitch_shift in range(0, 101, 10):
        bachpath = os.path.join(os.path.expanduser("~"), "violindataset", "Bach10-mf0-synth") \
                   + '_' + str(pitch_shift) + 'c_shifted'
        bach10_extract_pitch_with_model(new_model_name, bach10_path=bachpath, viterbi=False, verbose=1)

    # ViolinPedagogue
    extract_pitch_with_model(model_name=new_model_name,
                             main_dataset_folder=os.path.join(os.path.expanduser("~"),
                                                              "violindataset", "monophonic_etudes"),
                             save_activation=False, viterbi=True, verbose=0)

    # URMP
    urmp_all_instruments_extract_pitch_with_model(new_model_name, viterbi=False, verbose=1)

    '''
    external_data_extract_pitch_with_model(model_name=new_model_name,
                                           external_data_path=os.path.join(os.path.expanduser("~"),
                                                                           "violindataset", "monophonic_etudes",
                                                                           "allPaganini"),
                                           viterbi=True, verbose=1)



    bach10violineval = bach10_evaluate_all(instrument='violin')
    bach10eval = bach10_evaluate_all(pitch_range=(190, 4000))



    bach10_extract_pitch_with_model(new_model_name, viterbi=False, verbose=1)    
    '''

    # urmp_all_instruments_extract_pitch_with_model(new_model_name, viterbi=False, verbose=1)
    # urmp_evaluate_all(pitch_range=(190, 4000))

    '''
    viterbi = 'weird'
    extract_pitch_with_model(model_name=new_model_name,
                             main_dataset_folder=os.path.join(os.path.expanduser("~"),
                                                              "violindataset", "monophonic_etudes"),
                             save_activation=False, viterbi=viterbi, verbose=0)
    '''

    # for new_model_name in new_model_names:
    #    extract_pitch_with_model(model_name=new_model_name, save_activation=True, viterbi=True, verbose=0)

    # os.chdir(os.getcwd()+'/utils')
    # single_file_extract_pitch_with_model('kurdili_solo_violin.wav', model_name='iter1')
    # single_file_extract_pitch_with_model('kurdili_solo_violin.wav', model_name='original', viterbi='weird')

