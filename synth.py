import os
import numpy as np
import librosa
import pandas as pd
from scipy import interpolate
from scipy.signal import get_window, medfilt
from utils.pitchfilter import PitchFilter
import sys
import pickle
from pathlib import Path
if os.path.join(Path().absolute(), 'sms_tools', 'models') not in sys.path:
    sys.path.append(os.path.join(Path().absolute(), 'sms_tools', 'models'))
from sms_tools.models import hprModel as HPR
from sms_tools.models.harmonicModel import harmonicModelAnal
from sms_tools.models.utilFunctions import sineSubtraction
from sms_tools.models.stochasticModel import stochasticModelAnal
from sms_tools.models import sineModel as SM
from sms_tools.models.utilFunctions import refinef0Twm
import soundfile as sf
from joblib import Parallel, delayed
from sklearn.covariance import EllipticEnvelope
from time import time as taymit

HOP_SIZE = 128
SAMPLING_RATE = 44100
WINDOW_SIZE = 1001


def silence_segments_one_run(confidences, confidence_threshold, segment_len_th):
    conf_bool = np.array(confidences>confidence_threshold).reshape(-1)
    absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    segment_durs = np.diff(ranges,axis=1)
    valid_segments = ranges[np.repeat(segment_durs>segment_len_th, repeats=2, axis=1)].reshape(-1, 2)
    voiced = np.zeros(len(confidences), dtype=bool)
    for segment in valid_segments:
        voiced[segment[0]:segment[1]] = True
    return voiced


def silence_unvoiced_segments(pitch_track_csv, low_confidence_threshold=0.2,
                              high_confidence_threshold=0.7, min_voiced_segment_ms=12):
    """
    Accepts crepe output in the csv format and removes unvoiced segments with confidence and accepted voiced segment duration
    :param pitch_track_csv: csv with [ºtimeº, ºfrequencyº, ºconfidenceº] fields
    :param low_confidence_threshold: confidence threshold in range (0,1)
    :param high_confidence_threshold: confidence threshold in range (0,1)
    :param min_voiced_segment_ms: voiced segments shorter than the specified lenght are discarded
    :return: input csv file with the silenced segments
    """
    annotation_interval_ms = 1000*pitch_track_csv.loc[:1, "time"].diff()[1]
    voiced_th = int(np.ceil(min_voiced_segment_ms/annotation_interval_ms))

    # we do not accept the segment if a close neighbors do not have a confidence > 0.7
    smoothened_confidences = medfilt(pitch_track_csv["confidence"], kernel_size=2*(voiced_th//2)+1)
    smooth_voiced = silence_segments_one_run(smoothened_confidences,
                                             confidence_threshold=high_confidence_threshold, segment_len_th=voiced_th)

    # we also do not accept the pitch values if the individual confidences are really low
    hard_voiced = silence_segments_one_run(pitch_track_csv["confidence"],
                                           confidence_threshold=low_confidence_threshold, segment_len_th=voiced_th)
    
    # we accept the intersection of these two zones
    voiced = np.logical_and(smooth_voiced, hard_voiced)
    absdiff = np.abs(np.diff(np.concatenate(([False], voiced, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    unvoiced_ranges = np.vstack([ranges[:-1, 1], ranges[1:, 0]]).T
    for unvoiced_boundary in unvoiced_ranges:
        # we don't want small unvoiced zones. Check if they are acceptable with a more favorable mean thresholding
        if np.diff(unvoiced_boundary) < voiced_th:
            avg_convidence = pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"].mean()
            if avg_convidence > low_confidence_threshold:
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True

    pitch_track_csv.loc[~voiced, "frequency"] = 0
    return pitch_track_csv


def interpolate_f0_to_sr(pitch_track_csv, audio, sr=SAMPLING_RATE, hop_size=HOP_SIZE):
    f = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["frequency"],
                             kind="cubic", fill_value="extrapolate")
    c = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["confidence"],
                             kind="cubic", fill_value="extrapolate")
    start_frame = int(np.floor((WINDOW_SIZE + 1) / 2))
    end_frame = len(audio) - (len(audio) % hop_size) + start_frame
    time = np.array(range(start_frame, end_frame+1, hop_size)) / sr
    pitch_track_np = f(time)
    confidence_np = c(time)
    pitch_track_np[pitch_track_np < 10] = 0  # interpolation might introduce odd frequencies
    return pitch_track_np, confidence_np, time


def anal(audio, f0, hop_size=HOP_SIZE, sr=SAMPLING_RATE):
    # Get harmonic content from audio using extracted pitch as reference
    w = get_window('hanning', WINDOW_SIZE, fftbins=True)
    hfreq, hmag, hphase = harmonicModelAnal(
        x=audio,
        f0=f0,
        fs=sr,
        w=w,
        H=hop_size,
        N=2048,
        t=-90,
        nH=30,
        harmDevSlope=0.001,
        minSineDur=0.001
    )
    return hfreq, hmag, hphase


def refine_harmonics_twm(hfreq, hmag, hphases, f0, f0et=5.0, f0_refinement_range_cents=10, min_voiced_segment_ms=100):
    """
    Refine the f0 estimate with the help of two-way mismatch algorithm and change the harmonic components
    to the exact multiples of the refined f0 estimate
    :param hfreq: analyzed harmonic frequencies
    :param hmag: analyzed magnitudes
    :param f0: f0 in Hz before TWM
    :param f0et: error threshold for the TWM
    :param f0_refinement_range_cents: the range to be explored in TWM
    :return: new synthesis parameters
    """
    for frame, f0_frame in enumerate(f0):
        if f0_frame > 0:  # for the valid frequencies
            pfreq = hfreq[frame]
            pmag = hmag[frame]
            f0_twm, f0err_twm = refinef0Twm(pfreq, pmag, f0_frame, refinement_range_cents=f0_refinement_range_cents)
            if f0err_twm < f0et:
                hfreq[frame] = f0_twm * np.round(pfreq/f0_twm)
                f0[frame] = f0_twm
            else:
                f0[frame] = 0
                hfreq[frame] = 0
                hmag[frame] = 0
    
    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms/1000)/(HOP_SIZE/SAMPLING_RATE)))
    voiced = silence_segments_one_run(f0, 0, min_voiced_segment_len)
    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = 0
    hphases[~voiced] = 0
    return hfreq, hmag, hphases, f0


def supress_timbre_anomalies(instrument_detector, hfreq, hmag, hphase, f0):
    voiced = instrument_detector.predict(hmag[:, :12])
    voiced = voiced > 0
    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = 0
    hphase[~voiced] = 0
    return hfreq, hmag, hphase, f0


def apply_pitch_filter(pitch_track_csv, min_chunk_size=20, median=True, confidence_threshold=0.8):
    """AI is creating summary for apply_pitch_filter

    Args:
        pitch_track_csv (dataframe): time, frequency, confidence
        min_chunk_size (int, optional): The chunk size. Defaults to 20.
        median (bool, optional): Apply median filter or the filter from Baris. Defaults to True.
        confidence_threshold (float, optional): Apply filter only if confidence is smaller than the threshold value

    Returns:
        [type]: [description]
    """
    if median:
        # use filter when confidence is small
        filter_bool = np.array(pitch_track_csv["confidence"]<confidence_threshold).reshape(-1)
        # but in the voiced regions
        filter_bool = np.logical_and(filter_bool, np.array(pitch_track_csv["frequency"]>0).reshape(-1))
        # ensure the filter size is odd
        filtered_est = medfilt(pitch_track_csv["frequency"], kernel_size=(2*(min_chunk_size//2))+1)
        pitch_track_csv.loc[filter_bool, "frequency"] = filtered_est[filter_bool]
    else:
        pitch_filter = PitchFilter(min_chunk_size=min_chunk_size)
        pitch_track_np = pitch_filter.filter(np.array(pitch_track_csv))
        pitch_track_csv = pd.DataFrame(data=pitch_track_np, columns=pitch_track_csv.columns)
    return pitch_track_csv


def analyze_file(filename, path_folder_audio, path_folder_f0, path_folder_anal):
    time_start = taymit()
    audio = librosa.load(os.path.join(path_folder_audio, filename), sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(os.path.join(path_folder_f0, filename[:-3] + "f0.csv"))
    f0s, conf, time = interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load-time_start))
    hfreqs, hmags, _ = anal(audio, f0s)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()
    print("anal {:s} took {:.3f}".format(filename, time_anal-time_load))
    np.save(os.path.join(path_folder_anal, filename[:-3] + "npy"), hmags[conf > 0.9, :12])
    return


def analyze_folder(path_folder_audio, path_folder_f0, path_folder_anal, n_jobs=4):
    if not os.path.exists(path_folder_anal):
        # Create a new directory because it does not exist
        os.makedirs(path_folder_anal)
    Parallel(n_jobs=n_jobs)(delayed(analyze_file)(
        file, path_folder_audio, path_folder_f0, path_folder_anal) for file in sorted(os.listdir(path_folder_audio)))
    return


def process_file(filename, path_folder_audio, path_folder_f0, path_folder_synth, instrument_detector=None):
    th_lc = 0.2
    th_hc = 0.7
    voiced_th_ms = 100
    time_start = taymit()
    audio = librosa.load(os.path.join(path_folder_audio, filename), sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(os.path.join(path_folder_f0, filename[:-3] + "f0.csv"))
    pre_anal_coverage = f0s['confidence'] > th_lc
    pre_anal_coverage = sum(pre_anal_coverage)/len(pre_anal_coverage)
    f0s = silence_unvoiced_segments(f0s, low_confidence_threshold=th_lc, high_confidence_threshold=th_hc,
                                    min_voiced_segment_ms=voiced_th_ms)
    f0s = apply_pitch_filter(f0s, min_chunk_size=21, median=True, confidence_threshold=th_hc)
    f0s, conf, time = interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load-time_start))
    hfreqs, hmags, hphases = anal(audio, f0s)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()
    print("anal {:s} took {:.3f}".format(filename, time_anal-time_load))
    if instrument_detector is not None:
        hfreqs, hmags, hphases, f0 = supress_timbre_anomalies(instrument_detector, hfreqs, hmags, hphases, f0s)
    hfreqs, hmags, hphases, f0s = refine_harmonics_twm(hfreqs, hmags, hphases,
                                                       f0s, f0et=5.0, f0_refinement_range_cents=15,
                                                       min_voiced_segment_ms=voiced_th_ms)
    time_refine = taymit()
    post_anal_coverage = sum(f0s > 0) / len(f0s)
    coverage = post_anal_coverage / pre_anal_coverage
    print("refining parameters for {:s} took {:.3f}. coverage: {:.3f}".format(filename,
                                                                              time_refine-time_anal,
                                                                              coverage))
    harmonic_audio = SM.sineModelSynth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
    sf.write(os.path.join(path_folder_synth, filename[:-3] + "RESYN.wav"), harmonic_audio, 44100, 'PCM_24')
    df = pd.DataFrame([time, f0s]).T
    df.to_csv(os.path.join(path_folder_synth, filename[:-3] + "RESYN.csv"), header=False, index=False,
              float_format='%.6f')
    time_synth = taymit()
    print("synthesizing {:s} took {:.3f}. Total resynthesis took {:.3f}".format(filename, time_synth-time_refine,
                                                                                time_synth-time_load))
    return


def process_folder(path_folder_audio, path_folder_f0, path_folder_synth, instrument_detector=None,  n_jobs=4):
    if not os.path.exists(path_folder_synth):
        # Create a new directory because it does not exist 
        os.makedirs(path_folder_synth)
    Parallel(n_jobs=n_jobs)(delayed(process_file)(
        file, path_folder_audio, path_folder_f0, path_folder_synth, instrument_detector)
                            for file in sorted(os.listdir(path_folder_audio)))
    return


if __name__ == '__main__':
    names = ["L1", "L2", "L3", "L4", "L5", "L6"]
    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
    with open(os.path.join(dataset_folder, 'EllipticEnvelopeInstrumentModel.pkl'), 'rb') as modelfile:
        instrument_timbre_detector = pickle.load(modelfile)
    for name in names:
        time_grade = taymit()
        print("Started processing grade ", name)
        process_folder(path_folder_audio=os.path.join(dataset_folder, name),
                       path_folder_f0=os.path.join(dataset_folder, "pitch_tracks", "crepe_original", name),
                       path_folder_synth=os.path.join(dataset_folder, "synthesized", name),
                       instrument_detector=instrument_timbre_detector,
                       n_jobs=16)
        time_grade = taymit() - time_grade
        print("Grade {:s} took {:.3f}".format(name, time_grade))

