from synth import analyze_folder
import os
import pickle
import numpy as np
from sklearn.covariance import EllipticEnvelope

if __name__ == '__main__':
    names = ["L1", "L2", "L3", "L4", "L5", "L6"]
    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")
    for name in names:
        analyze_folder(path_folder_audio=os.path.join(dataset_folder, name),
                       path_folder_f0=os.path.join(dataset_folder, "pitch_tracks", "crepe_original", name),
                       path_folder_anal=os.path.join(dataset_folder, "anal", name),
                       n_jobs=16)

    data = []
    for name in names:
        print("started processing the folder ", name)
        path_folder_anal = os.path.join(dataset_folder, "anal", name)
        for file in sorted(os.listdir(path_folder_anal)):
            data.append(np.load(os.path.join(path_folder_anal, file)))
        data = np.vstack(data)
    model = EllipticEnvelope().fit(data)
    with open(os.path.join(dataset_folder, 'EllipticEnvelopeInstrumentModel.pkl'), 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

        """
        analyze_folder(path_folder_audio=os.path.join(dataset_folder, name),
                       path_folder_f0=os.path.join(dataset_folder, "pitch_tracks", "crepe_original", name),
                       path_folder_anal=os.path.join(dataset_folder, "anal", name),
                       n_jobs=16)
        """


