from hmmlearn import hmm
import numpy as np
from marl_crepe.core import to_local_average_cents
import os
import matplotlib.cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d




def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """


    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    #xx, yy = np.meshgrid(range(360), range(360))
    #transition = np.maximum(12 - abs(xx - yy), 0)
    #transition = transition / np.sum(transition, axis=1)[:, None]
    transition = gaussian_filter1d(np.eye(360), 30) + gaussian_filter1d(np.eye(360), 2)
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
                           range(len(observations))])


if __name__ == '__main__':
    plt.tight_layout(pad=0.05)
    #fig = plt.figure(figsize=)
    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'wspace': 0, 'hspace': 0.2}, squeeze=True, sharex=True)
    fig.set_size_inches([280, 7])
    for i, model_name in enumerate(['original', 'cleaned1200']):
        track = 'Kreutzer_BernardChevalier_030_Etude #30 violin by Rodolphe Kreutzer (1766-1831)'
        folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
        activations = os.path.join(folder, 'activations', model_name, 'L6', track + '.npy')
        activations = np.load(activations)
        p, c = to_viterbi_cents(activations)
        salience = np.flip(activations, axis=1)
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())
        #plt.imshow(image[:205,45:750,:])

        #axes.append(fig.add_subplot(2, 1, i + 1))
        axes[i].set_title(model_name)
        #axes[i].axis("off")
        axes[i].imshow(image[:205, 45:, :])
    #plt.gca().set_position((0, 0, 1, 1))
    plt.show()
    print('a')
