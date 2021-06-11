This is a repository containing the code behind the "Visualization and exploration of Spec2Vec mass spectral embeddings" research project.


List of dependencies to be able to run the code:

python 3.8.5 required

numpy (1.20+!)
 -> pip install numpy==1.20

seaborn
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


In the SubPositive folder, there is a .py script that generates 3 .png images, corresponding to PCA, t-SNE and UMAP results (respectively). Input required is a spectrum dataset, and a pre-trained model to create spectrum documents.

Compound_classes folder contains a .py script that retrieves classification information for compounds.

matchms_tutorial is a folder with notebooks to test the basic features of matchms and spec2vec.

Spec2vec_embedding_analysis.ipynb -> Notebook with the code necessary to reproduce the research. (previous output included)

Spec2vec_embedding_analysis_outputs_cleared.ipynb -> Same as above, but with outputs cleared.

Generate_SubPositive_Dataset.ipynb -> same as above, but simplified; only the code to run PCA, t-SNE and UMAP and plotting included. Code to select part of a plot and produce a table with spectrum metadata is absent!
