import copy

import librosa.sequence
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
    from librosa.sequence import viterbi_discriminative

    # transition probabilities inducing continuous pitch
    transition = gaussian_filter1d(np.eye(360), 30) + 10*gaussian_filter1d(np.eye(360), 2)
    transition = transition / np.sum(transition, axis=1)[:, None]

    p = copy.deepcopy(salience)
    p = p/p.sum(axis=1)[:, None]
    p[np.isnan(p.sum(axis=1)), :] = np.ones(360) * 1/360
    path = viterbi_discriminative(p.T, transition)

    return path, np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                           range(len(path))])


if __name__ == '__main__':

    for track in ['Kreutzer_BernardChevalier_030_Etude #30 violin by Rodolphe Kreutzer (1766-1831)',
                  'Kreutzer_BochanKang_029_Kreutzer Violin Etude No. 30 크로이쩌 바이올린 에튀드 30번 @ 보찬TV',
                  "Kreutzer_SunKim_030_Études ou caprices - No. 30 in B-Flat Major. Moderato"]:
        plt.tight_layout(pad=0.05)
        # fig = plt.figure(figsize=)
        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'wspace': 0, 'hspace': 0.2}, squeeze=True, sharex=True)
        fig.set_size_inches([180, 20])
        for i, model_name in enumerate(['original', 'iter1']):
            folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
            activations = os.path.join(folder, 'activations', model_name, 'L6', track + '.npy')
            activations = np.load(activations)
            p, c = to_viterbi_cents(activations)
            salience = np.flip(activations, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())
            #plt.imshow(image[:205,45:750,:])

            #axes.append(fig.add_subplot(2, 1, i + 1))
            axes[i].set_title(model_name+track)
            #axes[i].axis("off")
            axes[i].imshow(image[:205, 10000:15000, :])
        #plt.gca().set_position((0, 0, 1, 1))
        fig.show()
        print('a')
