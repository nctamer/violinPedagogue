import copy
import os
import numpy as np
import librosa
import pandas as pd
from scipy import interpolate
from scipy.signal import get_window, medfilt
from utils.pitchfilter import PitchFilter
import sys
import pickle
import random
from pathlib import Path

if os.path.join(Path().absolute(), 'sms_tools', 'models') not in sys.path:
    sys.path.append(os.path.join(Path().absolute(), 'sms_tools', 'models'))
from utils.synth2tfrecord import process_folder as synth2tfrecord_folder
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
WINDOW_SIZE = 1025  # int(2*(((1024/16000)*SAMPLING_RATE)//2))-1
WINDOW_TYPE = 'blackmanharris'
LOWEST_NOTE_ALLOWED_HZ = 180


def silence_segments_one_run(confidences, confidence_threshold, segment_len_th):
    conf_bool = np.array(confidences > confidence_threshold).reshape(-1)
    absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    segment_durs = np.diff(ranges, axis=1)
    valid_segments = ranges[np.repeat(segment_durs > segment_len_th, repeats=2, axis=1)].reshape(-1, 2)
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
    annotation_interval_ms = 1000 * pitch_track_csv.loc[:1, "time"].diff()[1]
    voiced_th = int(np.ceil(min_voiced_segment_ms / annotation_interval_ms))

    # we do not accept the segment if a close neighbors do not have a confidence > 0.7
    smoothened_confidences = medfilt(pitch_track_csv["confidence"], kernel_size=2 * (voiced_th // 2) + 1)
    smooth_voiced = silence_segments_one_run(smoothened_confidences,
                                             confidence_threshold=high_confidence_threshold, segment_len_th=voiced_th)

    # we also do not accept the pitch values if the individual confidences are really low
    hard_voiced = silence_segments_one_run(pitch_track_csv["confidence"],
                                           confidence_threshold=low_confidence_threshold, segment_len_th=voiced_th)

    # we accept the intersection of these two zones
    voiced = np.logical_and(smooth_voiced, hard_voiced)

    smoothened_pitch = copy.deepcopy(pitch_track_csv["frequency"])
    smoothened_pitch[~voiced] = np.nan
    smoothened_pitch.fillna(smoothened_pitch.rolling(window=15, min_periods=8).median(), inplace=True)

    # medfilt(pitch_track_csv["frequency"], kernel_size=21)
    absdiff = np.abs(np.diff(np.concatenate(([False], voiced, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    unvoiced_ranges = np.vstack([ranges[:-1, 1], ranges[1:, 0]]).T
    for unvoiced_boundary in unvoiced_ranges:
        # we don't want small unvoiced zones. Check if they are acceptable with a more favorable mean thresholding
        len_unvoiced = np.diff(unvoiced_boundary)[0]
        if len_unvoiced < voiced_th:
            avg_confidence = pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"].mean()
            if avg_confidence > low_confidence_threshold:
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
            elif len_unvoiced < 8:
                # and (unvoiced_boundary[0] > 3) and (unvoiced_boundary[-1] < len(pitch_track_csv)-3):
                pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] = \
                    smoothened_pitch[unvoiced_boundary[0]:unvoiced_boundary[1]]
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
                """
                past_voiced = pitch_track_csv.loc[unvoiced_boundary[0]-3:unvoiced_boundary[0]-1]
                next_voiced = pitch_track_csv.loc[unvoiced_boundary[1]:unvoiced_boundary[1]+2]
                past_conf = past_voiced["confidence"].mean()
                next_conf = next_voiced["confidence"].mean()
                if past_conf > next_conf:
                    if past_conf > low_confidence_threshold:
                        pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] =\
                            past_voiced["frequency"].median()
                        pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"] = past_conf
                        voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
                else:
                    if next_conf > low_confidence_threshold:
                        pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] =\
                            next_voiced["frequency"].median()
                        pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"] = next_conf
                        voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
                """

    pitch_track_csv.loc[~voiced, "frequency"] = 0
    pitch_track_csv["frequency"] = pitch_track_csv["frequency"].fillna(0)
    return pitch_track_csv


def interpolate_f0_to_sr(pitch_track_csv, audio, sr=SAMPLING_RATE, hop_size=HOP_SIZE):
    f = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["frequency"],
                             kind="nearest", fill_value="extrapolate")
    c = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["confidence"],
                             kind="nearest", fill_value="extrapolate")
    start_frame = 0  # I was true at first! It starts from zero!! int(np.floor((WINDOW_SIZE + 1) / 2))
    end_frame = len(audio) - (len(audio) % hop_size) + start_frame
    time = np.array(range(start_frame, end_frame + 1, hop_size)) / sr
    pitch_track_np = f(time)
    confidence_np = c(time)
    pitch_track_np[pitch_track_np < 10] = 0  # interpolation might introduce odd frequencies
    return pitch_track_np, confidence_np, time


def anal(audio, f0, n_harmonics=30, hop_size=HOP_SIZE, sr=SAMPLING_RATE):
    # Get harmonic content from audio using extracted pitch as reference
    w = get_window(WINDOW_TYPE, WINDOW_SIZE, fftbins=True)
    hfreq, hmag, hphase = harmonicModelAnal(
        x=audio,
        f0=f0,
        fs=sr,
        w=w,
        H=hop_size,
        N=2048,
        t=-90,
        nH=n_harmonics,
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
                hfreq[frame] = f0_twm * np.round(pfreq / f0_twm)
                f0[frame] = f0_twm
            else:
                f0[frame] = 0
                hfreq[frame] = 0
                hmag[frame] = -100

    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    voiced = silence_segments_one_run(f0, 0, min_voiced_segment_len)
    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
    hphases[~voiced] = 0
    return hfreq, hmag, hphases, f0


def supress_timbre_anomalies(instrument_detector, hfreq, hmag, hphase, f0):
    hmag_ptr = np.copy(hmag)[:, 1:12]
    hmag_ptr[np.isnan(hmag_ptr)] = -100
    hvalid = np.logical_and(hmag_ptr[:, 0] > -100, f0 > 0)
    hmag_ptr[hvalid] = hmag_ptr[hvalid] - hmag[hvalid, 0][:, np.newaxis]
    filter_centroids = np.array(list(instrument_detector.keys()))
    f0_classes = np.abs(f0[:,np.newaxis] - filter_centroids[np.newaxis,:]).argmin(axis=1)
    relevant_classes = np.unique(f0_classes[hvalid])
    voiced = hvalid
    for pitch_class in relevant_classes:
        relevant_timbre_model = instrument_detector[filter_centroids[pitch_class]]
        relevant_section = np.logical_and(f0_classes==pitch_class, hvalid)
        voiced_relevant = relevant_timbre_model.predict(hmag_ptr[relevant_section])
        voiced[relevant_section] = voiced_relevant > 0

    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
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
        filter_bool = np.array(pitch_track_csv["confidence"] < confidence_threshold).reshape(-1)
        # but in the voiced regions
        filter_bool = np.logical_and(filter_bool, np.array(pitch_track_csv["frequency"] > 0).reshape(-1))
        # ensure the filter size is odd
        filtered_est = medfilt(pitch_track_csv["frequency"], kernel_size=(2 * (min_chunk_size // 2)) + 1)
        pitch_track_csv.loc[filter_bool, "frequency"] = filtered_est[filter_bool]
    else:
        pitch_filter = PitchFilter(min_chunk_size=min_chunk_size)
        pitch_track_np = pitch_filter.filter(np.array(pitch_track_csv))
        pitch_track_csv = pd.DataFrame(data=pitch_track_np, columns=pitch_track_csv.columns)
    return pitch_track_csv


def analyze_file(filename, path_folder_audio, path_folder_f0, path_folder_anal, confidence_threshold=0.9,
                 min_voiced_segment_ms=25):
    time_start = taymit()
    audio = librosa.load(os.path.join(path_folder_audio, filename), sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(os.path.join(path_folder_f0, filename[:-3] + "f0.csv"))
    f0s, conf, time = interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load - time_start))
    hfreqs, hmags, _ = anal(audio, f0s, n_harmonics=12)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()

    conf_bool = conf > confidence_threshold
    conf_bool_1 = conf < 1.0
    conf_bool = np.logical_and(conf_bool, conf_bool_1)
    valid_f0_bool = f0s > LOWEST_NOTE_ALLOWED_HZ
    # lowest note on violin is G3 = 196 hz, so threshold with sth close to the lowest note
    valid_hmag_bool = (hmags > -100).sum(axis=1) > 3  # at least three harmonics
    valid_bool = np.logical_and(valid_hmag_bool, valid_f0_bool)
    valid_bool = np.logical_and(conf_bool, valid_bool)
    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    valid_bool = silence_segments_one_run(valid_bool, 0, min_voiced_segment_len)  # if keeps high for some duration

    print("anal {:s} took {:.3f}. coverage: {:.3f}".format(filename, time_anal - time_load,
                                                           sum(valid_bool) / len(valid_bool)))
    np.savez_compressed(os.path.join(path_folder_anal, filename[:-3] + "npz"),
                        f0=f0s[valid_bool], hmag=hmags[valid_bool, :12])
    return


def analyze_folder(path_folder_audio, path_folder_f0, path_folder_anal, confidence_threshold=0.9, n_jobs=4):
    if not os.path.exists(path_folder_anal):
        # Create a new directory because it does not exist
        os.makedirs(path_folder_anal)
    Parallel(n_jobs=n_jobs)(delayed(analyze_file)(
        a_file, path_folder_audio, path_folder_f0, path_folder_anal, confidence_threshold=confidence_threshold) for
                            a_file in sorted(os.listdir(path_folder_audio)))
    return


def process_file(filename, path_folder_audio, path_folder_f0, path_folder_synth,
                 instrument_detector=None, refine_twm=True, pitch_shift=False,
                 th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False):
    time_start = taymit()
    audio = librosa.load(os.path.join(path_folder_audio, filename), sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(os.path.join(path_folder_f0, filename[:-3] + "f0.csv"))
    f0s["confidence"] = f0s["confidence"].fillna(0)
    pre_anal_coverage = f0s['confidence'] > th_lc
    pre_anal_coverage = sum(pre_anal_coverage) / len(pre_anal_coverage)
    f0s = silence_unvoiced_segments(f0s, low_confidence_threshold=th_lc, high_confidence_threshold=th_hc,
                                    min_voiced_segment_ms=voiced_th_ms)
    # f0s = apply_pitch_filter(f0s, min_chunk_size=21, median=True, confidence_threshold=th_hc)
    f0s, conf, time = interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load - time_start))
    hfreqs, hmags, hphases = anal(audio, f0s, n_harmonics=40)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()
    print("anal {:s} took {:.3f}".format(filename, time_anal - time_load))
    if instrument_detector is not None:
        hfreqs, hmags, hphases, f0 = supress_timbre_anomalies(instrument_detector, hfreqs, hmags, hphases, f0s)
        f0et = 10.0
    else:
        f0et = 5.0
    if refine_twm:
        hfreqs, hmags, hphases, f0s = refine_harmonics_twm(hfreqs, hmags, hphases,
                                                           f0s, f0et=f0et, f0_refinement_range_cents=16,
                                                           min_voiced_segment_ms=voiced_th_ms)
    time_refine = taymit()
    post_anal_coverage = sum(f0s > 0) / len(f0s)
    coverage = post_anal_coverage / pre_anal_coverage
    print("refining parameters for {:s} took {:.3f}. coverage: {:.3f}".format(filename,
                                                                              time_refine - time_anal,
                                                                              coverage))
    if sawtooth_synth:
        hmags[f0s > 0] = -30 - 20 * np.log10(np.arange(1, 41))
        hfreqs[f0s > 0] = np.dot(hfreqs[f0s > 0][:, 0][:, np.newaxis], np.arange(1, 41)[np.newaxis, :])
        hphases = np.array([])
    harmonic_audio = SM.sineModelSynth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
    sf.write(os.path.join(path_folder_synth, filename[:-3] + "RESYN.wav"), harmonic_audio, 44100, 'PCM_24')
    df = pd.DataFrame([time, f0s]).T
    df.to_csv(os.path.join(path_folder_synth, filename[:-3] + "RESYN.csv"), header=False, index=False,
              float_format='%.6f')
    if pitch_shift:
        sign = random.choice([-1, 1])
        val = random.choice(range(5, 50))
        pitch_shift_cents = sign * val

        alt_f0s = f0s * pow(2, (pitch_shift_cents / 1200))
        # Synthesize audio with the shifted harmonic content
        alt_hfreqs = hfreqs * pow(2, (pitch_shift_cents / 1200))
        alt_harmonic_audio = SM.sineModelSynth(alt_hfreqs, hmags, np.array([]), N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
        sf.write(os.path.join(path_folder_synth, filename[:-3] + "shiftedRESYN.wav"), alt_harmonic_audio,
                 44100, 'PCM_24')
        df = pd.DataFrame([time, alt_f0s]).T
        df.to_csv(os.path.join(path_folder_synth, filename[:-3] + "shiftedRESYN.csv"), header=False, index=False,
                  float_format='%.6f')

    time_synth = taymit()
    print("synthesizing {:s} took {:.3f}. Total resynthesis took {:.3f}".format(filename, time_synth - time_refine,
                                                                                time_synth - time_load))
    return


def process_folder(path_folder_audio, path_folder_f0, path_folder_synth, pitch_shift=False,
                   instrument_detector=None, refine_twm=True,
                   th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, n_jobs=4):
    if not os.path.exists(path_folder_synth):
        # Create a new directory because it does not exist
        os.makedirs(path_folder_synth)
    Parallel(n_jobs=n_jobs)(delayed(process_file)(
        pr_file, path_folder_audio, path_folder_f0, path_folder_synth, pitch_shift=pitch_shift,
        instrument_detector=instrument_detector,
        th_lc=th_lc, th_hc=th_hc, voiced_th_ms=voiced_th_ms, refine_twm=refine_twm, sawtooth_synth=sawtooth_synth)
                            for pr_file in sorted(os.listdir(path_folder_audio)))
    return


if __name__ == '__main__':

    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "monophonic_etudes")
    names = sorted([_ for _ in os.listdir(dataset_folder) if (_.startswith('L') or _.startswith('mono'))])
    model = "original"
    mode_ablation = False  # only use for the ablation study *standard analysis-synthesis without the instrument model

    if not mode_ablation:
        # Instrument model is used for the standard implementation, below is the code to create the
        # instrument timbre model
        num_filters = 50
        contamination = 0.05
        instrument_model_method = "normalized"
        estimate_instrument_model = True
        inst_model_use_existing_anal_files = True  # todo just a workaround for the memory leak!

        if estimate_instrument_model:
            print("started instrument model estimation")
            # combine instrument model estimation with the synthesis. The analysis for the instrument estimation takes
            # a long while, so only do it when really needed!
            if not inst_model_use_existing_anal_files:
                for name in names:
                    analyze_folder(path_folder_audio=os.path.join(dataset_folder, name),
                                   path_folder_f0=os.path.join(dataset_folder, "pitch_tracks", model, name),
                                   path_folder_anal=os.path.join(dataset_folder, "anal", name),
                                   confidence_threshold=0.9,
                                   n_jobs=16)
            data, pitch_content = [], []
            for name in names:
                print("started processing the folder ", name)
                files_anal = os.path.join(dataset_folder, "anal", name)
                for file in sorted(os.listdir(files_anal)):
                    file_content = np.load(os.path.join(files_anal, file))
                    data.append(file_content['hmag'])
                    pitch_content.append(file_content['f0'])

            data = np.vstack(data)
            pitch_content = np.hstack(pitch_content)
            order = pitch_content.argsort()
            pitch_content = pitch_content[order]
            data = data[order]
            data = data - data[:, 0][:, np.newaxis]

            mids = np.linspace(pitch_content.min(), pitch_content.max(), num_filters + 2)
            increment = np.diff(mids).mean()
            instrument_timbre_detectors = {}
            print('range:', mids.min(), mids.max())
            for n in range(1, num_filters + 1):
                mid = mids[n]
                start = mid - increment
                end = mid + increment
                relevant = np.logical_and(pitch_content > start, pitch_content < end)
                relevant_data = data[relevant]
                instrument_timbre_detector = EllipticEnvelope(contamination=contamination).fit(relevant_data[:, 1:])
                instrument_timbre_detectors[mid] = instrument_timbre_detector
                print(mid, n, start, end, len(relevant_data))
                print(np.array2string(instrument_timbre_detector.location_, precision=2))

            pitch_bins = mids[1:-2]
            pitch_hist, _ = np.histogram(pitch_content, bins=pitch_bins)
            pitch_dist = pitch_hist / len(data)

            print("Pitch distribution for the instrument model:")
            for f, p in zip(pitch_bins, pitch_dist):
                print("%4d" % f, "*" * int(p * 100))

            with open(os.path.join(dataset_folder, 'instrument_model_' + str(num_filters) + '_' +
                                                   str(contamination) + '.pkl'), 'wb') as outp:
                pickle.dump(instrument_timbre_detectors, outp, pickle.HIGHEST_PROTOCOL)
            print("FINISHED INSTRUMENT MODEL ESTIMATION!!! \n\n\n\n\n\n\n\n NOW THE SYNTHESIS STARTS!!!")

        instrument_model_file = os.path.join(dataset_folder, 'instrument_model_' + str(num_filters) + '_' +
                                             str(contamination) + '.pkl')
        with open(instrument_model_file, 'rb') as modelfile:
            instrument_timbre_detector = pickle.load(modelfile)

        low_confidence_threshold = 0.2
        high_confidence_threshold = 0.6
        min_voiced_th_ms = 100
        refine_estimates_with_twm = True
        create_pitch_shifted_versions = True
        use_sawtooth_timbre = False

    else:  # for the ablation study, simple analysis-synthesis described in Salomon paper.
        instrument_timbre_detector = None
        low_confidence_threshold = 0.3
        high_confidence_threshold = 0.7
        min_voiced_th_ms = 50
        refine_estimates_with_twm = True
        create_pitch_shifted_versions = False
        use_sawtooth_timbre = True


    for name in sorted(names)[::-1]:
        time_grade = taymit()
        print("Started processing grade ", name)
        process_folder(path_folder_audio=os.path.join(dataset_folder, name),
                       path_folder_f0=os.path.join(dataset_folder, "pitch_tracks", model, name),
                       path_folder_synth=os.path.join(dataset_folder, "synthesized_instrument_model_" +
                                                      str(num_filters) + '_' + str(contamination), name),
                       instrument_detector=instrument_timbre_detector,
                       pitch_shift=create_pitch_shifted_versions,
                       th_lc=low_confidence_threshold, th_hc=high_confidence_threshold,
                       voiced_th_ms=min_voiced_th_ms, refine_twm=refine_estimates_with_twm,
                       sawtooth_synth=use_sawtooth_timbre, n_jobs=16)
        synth2tfrecord_folder(path_folder_synth=os.path.join(dataset_folder, "synthesized_instrument_model_" +
                                                      str(num_filters) + '_' + str(contamination), name),
                              path_folder_tfrecord=os.path.join(dataset_folder, "tfrecord_instrument_model_" +
                                                      str(num_filters) + '_' + str(contamination), name),
                              n_jobs=16)
        time_grade = taymit() - time_grade
        print("Grade {:s} took {:.3f}".format(name, time_grade))
