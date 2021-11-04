#29/10/2021 by Artur van Bemmelen
#I wrote this script to convert GNPS spectra to MS2DeepScore vectors with paralel computing.

import pickle
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2deepscore import SpectrumBinner

data_dir = "/mnt/scratch/vanb001/spec2vec/"
spectra = pickle.load(open(data_dir+"ALL_GNPS_210409_positive_cleaned_peaks_processed_fingerprints_added_s2v.pickle", "rb"))

model = load_model(data_dir+"ms2ds_model_20210419-221701_data210409_10k_500_500_200.hdf5")
ms2ds_score = MS2DeepScore(model)

def calculate_ms2ds_embedding(spec):
    binned_spec = model.spectrum_binner.transform([spec], progress_bar=False)[0]
    embed = model.base.predict(ms2ds_score._create_input_vector(binned_spec))
    return embed

embeddings = Parallel(n_jobs=24)(delayed(calculate_ms2ds_embedding)(spectra[i]) for i in range(len(spectra))) # Change n_jobs to whatever value suits best.

with open(data_dir+"MS2DeepScore_embedding_ALL_GNPS_210409_positive.pickle", 'wb') as f:
    pickle.dump(embeddings, f)
