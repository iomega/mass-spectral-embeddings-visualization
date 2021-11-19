# Content description
This is a repository containing the code behind the "Visualization and exploration of Spec2Vec mass spectral embeddings" research project by Antonio Rebac, as well as data explorations by Artur van Bemmelen.

```
List of dependencies to be able to run the code:

python 3.8.5 required

 numpy (1.20+!)
 -> pip install numpy==1.20

>seaborn
    -> pip install seaborn

sklearn
    -> pip install sklearn

pynndescent
    -> pip install pynndescent

spec2vec
    -> conda install --channel nlesc --channel bioconda --channel conda-forge spec2vec

matchms
    -> conda install --channel bioconda --channel conda-forge matchms

levenshtein
    -> pip install levenshtein
```

## Summary work of Antonio Rebac
In the SubPositive folder, there is a .py script that generates 3 .png images, corresponding to PCA, t-SNE and UMAP results (respectively). Input required is a spectrum dataset, and a pre-trained model to create spectrum documents.

Compound_classes folder contains a .py script that retrieves classification information for compounds.

matchms_tutorial is a folder with notebooks to test the basic features of matchms and spec2vec.

Spec2vec_embedding_analysis.ipynb -> Notebook with the code necessary to reproduce the research. (previous output included)

Spec2vec_embedding_analysis_outputs_cleared.ipynb -> Same as above, but with outputs cleared.

Generate_SubPositive_Dataset.ipynb -> same as above, but simplified; only the code to run PCA, t-SNE and UMAP and plotting included. Code to select part of a plot and produce a table with spectrum metadata is absent!

## Summary work of Artur van Bemmelen
The complete version of the summary is available in this Git directory as Word document. This document also contains a data management section, detailing the inputs neccessary to replicate the work, and the notable outputs of each script.

### t-SNE and UMAP comparison.ipynb**

A first glance at UMAP and t-SNE, in particular a t-SNE produced by Joris Louwen. The UMAPs were made with only 15 neighbours. Later work suggests that 50 neighbors is a much better starting point for a dataset of this size and complexity. The best feature of this notebook is the code for “coupled” interactive plots, where two or more interactive plots are controlled with the same legend. 


### Instrument type comparison.ipynb

This notebook takes a look at how spectra with the same planar inchikey that are measured by different instrument types, are embedded in t-SNE. I distinguish between HCD, CID, qToF, and Orbitrap. The embeddings of HCD Velos and CID Velos instruments are strikingly similar in my t-SNE embeddings, but differ in the 21/7 t-SNE embedding by Joris Louwen. 


### Jaccard similarity per classification group.ipynb

To gain an understanding of which groups may be expected to cluster together in embeddings, I computed the internal Jaccard similarity per class and classifier. As was to be expected, classification at subclass level resulted in classes with more internal similarity, whereas internal similarity decreased at superclass level. NPClassifier classes stood out as the classifier and class level where most spectra were classified into “high similarity” classes. 
Internal jaccard similarity per classification group.csv


### Jaccard similarity between groups.ipynb

To complement the notebook ‘Jaccard similarity per classification group’, I computed the similarity between classes of the same classifier to quantify the distinctness of each class. As was expected, the between-class similarity of Classyfire was lower than NPClassifier, as NPClassifier was designed specifically for natural products which would result less distinct classes. The similarity matrices this notebook output could elucidate confusion matrices of similarity-based classification algorithms.
Jaccard similarity between NPClassifier classes.csv
Jaccard similarity between Classyfire classes.csv


### Spec2Vec similarity per classification group.ipynb

To quantify how well the parameters for the Spec2Vec embedding process were chosen, and how much of the internal similarity the Spec2Vec embeddings capture, the Spec2Vec similarity of each class and classifier was also computed. The results show that there is much space for improvement, which, while disappointing, are not surprising considering Spec2Vec’s tendency to underestimate the similarity of all compounds with a Jaccard score lower than ~0.8. 
Internal spec2vec similarity per classification group.csv


### Compare internal similarities to MS2DeepScore.ipynb

After determining the internal similarities of various Classyfire classes and NPClassifier classes using Jaccard score and Spec2Vec similarity, the point was raised that it would be very interesting to see how these compare to the supervised embeddings produced by MS2DeepScore. Calculating the embeddings takes a while, so I wrote a script that uses parallel computation to speed up the process. Whereas Spec2Vec drastically underestimates the similarity of all but highly similar compounds, MS2DeepScore slightly overestimates the similarity. Setting the cut-off value for “high similarity” just slightly higher (0.65 instead of 0.6) however, created a nice correlation with the percentage of pairs exceeding 0.6 Jaccard similarity. This was true for both NPClassifier classes, and Classyfire classes, suggesting this threshold might generalize to other datasets.
Calculate_MS2DS_embeddings.py (available on Github)
MS2DeepScore_embedding_annotated_spectra_210409_joblib.pickle


### Unsupervised spec2vec UMAP exploration.ipynb

Spec2Vec-based UMAP exploration of classes larger than 3000 spectra of each of the following classification groups: NPClassifier pathway, NPClassifier superclass, Classyfire superclass, and Classyfire class. It includes minimal working examples for static seaborn plots and interactively plots using plotly.express.
spec2vec_npc_pathway_umap.pickle
spec2vec_npc_superclass_umap.pickle
spec2vec_cf_superclass_umap.pickle
spec2vec_cf_class_umap.pickle


### Unsupervised MS2DS UMAP exploration.ipynb

MS2DeepScore-based UMAP exploration of classes larger than 3000 spectra of each of the following classification groups: NPClassifier pathway, NPClassifier superclass, Classyfire superclass, and Classyfire class.
ms2ds_npc_pathway_umap.pickle
ms2ds_npc_superclass_umap.pickle
ms2ds_cf_superclass_umap.pickle
ms2ds_cf_class_umap.pickle


### Supervised UMAP – minimal working examples.ipynb

This notebook contains minimal working examples for a few methods of supervised UMAP clustering. Please note that these results are not meant to be exhaustive, methods that appear to perform poorly may yet yield surprising results with different parameters. The methods outlined in this notebook are: supervised UMAP, semi-supervised UMAP, and tree-based UMAP using ExtraTrees. To illustrate how different tree-based UMAPs can look, even when using the same model, UMAPs of the best and worst ExtraTrees model are shown.

