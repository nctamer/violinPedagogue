import os

from flazy import Dataset
from mir_eval.melody import hz2cents
from scipy.stats import norm

from transforms import *

classifier_lowest_hz = 31.70
classifier_lowest_cent = hz2cents(np.array([classifier_lowest_hz]))[0]
classifier_cents_per_bin = 20
classifier_octaves = 6
classifier_total_bins = int((1200 / classifier_cents_per_bin) * classifier_octaves)
classifier_cents = np.linspace(0, (classifier_total_bins - 1) * classifier_cents_per_bin, classifier_total_bins) + classifier_lowest_cent
classifier_cents_2d = np.expand_dims(classifier_cents, axis=1)
classifier_norm_stdev = 25
classifier_pdf_normalizer = norm.pdf(0)


def to_classifier_label(pitch):
    """
    Converts pitch labels in cents, to a vector representing the classification label
    Uses the normal distribution centered at the pitch and the standard deviation of 25 cents,
    normalized so that the exact prediction has the value 1.0.
    :param pitch: a number or numpy array of shape (1,)
    pitch values in cents, as returned by hz2cents with base_frequency = 10 (default)
    :return: ndarray
    """
    result = norm.pdf((classifier_cents - pitch) / classifier_norm_stdev).astype(np.float32)
    result /= classifier_pdf_normalizer
    return result


def to_weighted_average_cents(label):
    if label.ndim == 1:
        productsum = np.sum(label * classifier_cents)
        weightsum = np.sum(label)
        return productsum / weightsum
    if label.ndim == 2:
        productsum = np.dot(label, classifier_cents)
        weightsum = np.sum(label, axis=1)
        return productsum / weightsum
    raise Exception("label should be either 1d or 2d ndarray")


def to_local_average_cents(salience, center=None):
    """find the weighted average cents near the argmax bin"""

    import numpy as np

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """Find the Viterbi path using a transition prior that induces pitch continuity"""

    import numpy as np
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the others
    self_emission = 0.1
    emission = np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 359)

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_ = starting
    model.transmat_ = transition
    model.emissionprob_ = emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])


def train_dataset(*names, batch_size=32, loop=True, augment=True) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    # LAST ONE FOR THE PARENT FOLDER
    paths = [j for i in names for j in i]  # join separate train paths (list of lists -> single list)

    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

    if loop:
        datasets = [dataset.repeat() for dataset in datasets]

    result = Dataset.roundrobin(datasets)
    result = result.starmap(normalize)

    if augment:
        result = result.starmap(add_noise)
        result = result.starmap(pitch_shift)

    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))

    if batch_size:
        result = result.batch(batch_size)

    return result


def validation_dataset(*names, seed=None, take=None) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    all_datasets = []

    for files in names:
        if seed:
            files = Random(seed).sample(files, len(files))

        datasets = [Dataset.read.tfrecord(file, compression='gzip') for file in files]
        datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

        if seed:
            datasets = [dataset.shuffle(seed=seed) for dataset in datasets]
        if take:
            datasets = [dataset.take(take) for dataset in datasets]

        all_datasets.append(Dataset.concat(datasets))

    result = Dataset.roundrobin(all_datasets)
    result = result.starmap(normalize)
    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))

    return result
