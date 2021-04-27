#!/usr/bin/env python3

'''The purpose of this script is to load a .pickle file containing
unprocessed MS2 spectra, filter them and make a subset
of desired size, ultimately performing PCA, t-SNE and UMAP and plotting the
results in a scatter plot and grouping(coloring) according to inchikeys.
'''

from __future__ import print_function
import os
import sys
import gensim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import add_losses
from typing import Union
import numba
from gensim.models.basemodel import BaseTopicModel
from spec2vec.Document import Document
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from pca import pca
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import time


#function definitions

def load_pickle_file(pickle_filename, datafolder_path):
    '''Locates .pickle file and loads it.
    
    Input:
        pickle_filename -> STR, name of file
        datafolder_path -> STR, absolute path to folder where .pickle file is
                           located
    Returns:
        spectra -> LIST, loaded list of spectra from pickle file
    '''
    outfile = os.path.join(datafolder_path, pickle_filename)
    with open(outfile, 'rb') as filename:
        spectra = pickle.load(filename)
    return spectra

def post_process_s2v(s):
    '''Applies post processing steps to a spectrum (filtering).
    
    Input:
        s -> matchms.Spectrum.Spectrum object, a single spectrum
    Returns:
        s -> matchms.Spectrum.Spectrum object, a single processed spectrum
    '''
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=10)
    s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5)
    if s is None:
        return None
    s_remove_low_peaks = select_by_relative_intensity(s, intensity_from=0.001)
    if len(s_remove_low_peaks.peaks) >= 10:
        s = s_remove_low_peaks
    s = add_losses(s, loss_mz_from=5.0, loss_mz_to=200.0)
    return s

def build_inchikey_series(s2v_spectra):
    '''Builds a Pandas series of inchikeys and their frequency in dataset.
    
    Input:
        s2v_spectra -> LIST, spectra filtered by "post_process_s2v"
    Returns:
        inchikeys_series -> Pandas series, inchikeys and their frequency
        inchikeys_list   -> LIST, list of inchikeys for each spectrum in
                            s2v_spectra
    '''
    inchikeys_list = []
    for spectrum in s2v_spectra:
        inchikeys_list.append(spectrum.get("inchikey"))
    #make Pandas series excluding empty ones
    inchikeys_series = pd.Series([x for x in inchikeys_list if x])
    return inchikeys_series, inchikeys_list
    
def inchikey_selection(min_copies_in_data, inchikeys_series):
    '''Makes a sorted Pandas dataframe of inchikeys occurring n times.
    
    Input:
        min_copies_in_data -> INT, min amount of times an inchikey should occur
        inchikeys_series -> Pandas series, inchikeys and their frequency
    Returns:
        suitable_values -> Pandas dataframe; contains inchikeys occurring a
                           specified amount of times, and their frequencies in
                           a separate column
    '''
    if not isinstance(min_copies_in_data, int):
        raise ValueError("Input must be an INTEGER.")
    if min_copies_in_data > inchikeys_series.str[:14].value_counts()[0]:
        raise ValueError("Input cannot be larger than {}.".format\
            (inchikeys_pdframe.str[:14].value_counts()[0]))
    suitable_values = pd.DataFrame(inchikeys_series.str[:14].value_counts()\
        [inchikeys_series.str[:14].value_counts().values >=\
        min_copies_in_data])
    suitable_values.reset_index(level=suitable_values.index.names,\
        inplace=True)
    suitable_values.columns = (['inchikey14', 'occurences'])
    #sort values to make it reproducible
    suitable_values = suitable_values.sort_values(['occurences',\
        'inchikey14'], ascending=False)
    return suitable_values

def random_selection(pd_dataframe, quantity, seed=35):
    '''Randomly selects a given amount of inchikeys from a Pandas dataframe.
    
    Input:
        pd_dataframe -> Pandas dataframe; 2 columns: values and their
                           frequencies
        quantity -> INT, amount of values to be randomly selected
        seed -> INT, seed number to make the results reproducible (default 35)
    Returns:
        selected_values -> numpy array, contains a specified amount of randomly
                           selected values from pd_dataframe
    '''
    if not isinstance(quantity, int):
        raise ValueError("Input must be an INTEGER.")
    if quantity > pd_dataframe.shape[0]:
        raise ValueError("Input cannot be larger than {}.".format\
            (pd_dataframe.shape[0]))
    np.random.seed(seed)
    selection = np.random.choice(pd_dataframe.shape[0], quantity,\
        replace=False)
    selected_values = pd_dataframe['inchikey14'].values[selection]
    return selected_values

def find_spectra(selected_values, inchikeys_list, seed=35):
    '''Finds matching spectra for selected inchikeys and stores them in a list.
    
    Input:
        selected_values -> numpy array, contains a specified amount of randomly
                           selected inchikeys
        inchikeys_list -> LIST, list of all inchikeys of imported dataset
        seed -> INT, seed number to make the results reproducible (default 35)
    Returns:
        matching_spectra -> LIST, contains the spectra represented by the
                            previously randomly selected inchikeys
    '''
    matching_spectra = []
    inchikeys_pd = pd.Series([x for x in inchikeys_list]) #include empty ones
    np.random.seed(seed) #to make reproducible
    for inchikey in selected_values:
        matches = inchikeys_pd[inchikeys_pd.str[:14] == inchikey].index.values
        matching_spectra.extend(matches)
    return matching_spectra
    
def load_trained_model(filename, path_data):
    '''Loads a pre-trained spec2vec model.
    
    Input:
        filename -> STR, name of the model file
        path_data -> STR, absolute path to folder where the .model file is
    Returns:
        trained_model -> gensim.models.word2vec.Word2Vec, variable containing
                         the model
    '''
    if filename.endswith(".model"):
        filename_model = os.path.join(path_data, filename)
        trained_model = gensim.models.Word2Vec.load(filename_model)
    else:
        raise ValueError("Trained spec2vec model name should end in '.model'")
    return trained_model

def find_vectors(spec_docs, trained_model):
    '''Calculates vectors using trained model and stores them in numpy array.
    
    Input:
        spec_docs -> LIST, contains relevant spectrum information
        trained_model -> gensim.models.word2vec.Word2Vec, variable containing
                         the model
    Returns:
        spec_vectors -> numpy array, contains calculated vectors
    '''
    list_of_vectors = []
    for doc in spec_docs:
        single_vector = calc_vector(trained_model, doc)
        list_of_vectors.append(single_vector)
    spec_vectors = np.stack(list_of_vectors)
    return spec_vectors

def make_inchikey_list(s2v_spectra, selected_spectrums):
    '''Makes an ordered list of inchikeys.
    
    Input:
        s2v_spectra -> LIST, spectra filtered by "post_process_s2v"
        selected_spectrums -> LIST, spectra that went under selection process
                              in "find_spectra"
    Returns:
        inchikey_list -> LIST, inchikeys in the same order as the vector list
    '''
    inchikey_list = []
    for i, s in enumerate(s2v_spectra):
        if i in selected_spectrums:
            inchikey_list.append(s.metadata.get("inchikey")[:14])
    return inchikey_list

def make_dataset(spec_vectors, inchikey_list):
    '''Creates a Pandas dataframe ready to plot with.
    
    Input:
        spec_vectors -> numpy array, contains calculated vectors
        inchikey_list -> LIST, inchikeys in the same order as the vector list
    Returns:
        dataset -> Pandas dataframe, contains spec2vec vectors, uses inchikeys
                   as index
    '''
    dataset = pd.DataFrame(spec_vectors, index=inchikey_list,\
        columns = [str(i) for i in range(spec_vectors.shape[1])])
    return dataset

def perform_pca(dataset):
    '''Performs PCA on a Pandas dataframe with spec2vec vectors.
    
    Input:
        dataset -> Pandas dataframe, contains spec2vec vectors, uses inchikeys
                   as index
    Returns:
        pca_model -> pca.pca.pca class
        pca_results -> DICT, contains descriptive PCA results
    '''
    pca_model = pca(n_components=3)
    pca_results = pca_model.fit_transform(dataset)
    return pca_model, pca_results

def plot_pca_2D(pca_model, pickle_file, amount, occurrences):
    '''Makes a 2D scatterplot based on PCA results and stores it in a .png.
    
    Input:
        pca_model -> pca.pca.pca class
        pickle_file -> STR, name of .pickle file containing the spectra
        amount -> INT, amount of inchikeys randomly selected
        occurrences -> INT, min amount of inchikeys that should occur
    Returns:
        outfile -> STR, name of output file containing the plot
    '''
    fig, ax = pca_model.scatter()
    outfile = "PCA_2D_scatter_of_{}_inchikeys_occurring_{}_times_in_{}.png"\
        .format(amount, occurrences, pickle_file)
    if os.path.exists(outfile) == False:
        fig.savefig(outfile)
        print("PCA plot stored in file: {}".format(outfile))
    else:
        print("WARNING! File name already exists, new plot will not be made!")
        print("File name: {}".format(outfile))
    return outfile
    
def plot_pca_3D(pca_model, pickle_file, amount, occurrences):
    '''Makes a 3D scatterplot based on PCA results and stores it in a .png.
    
    Input:
        pca_model -> pca.pca.pca class
        pickle_file -> STR, name of .pickle file containing the spectra
        amount -> INT, amount of inchikeys randomly selected
        occurrences -> INT, min amount of inchikeys that should occur
    Returns:
        outfile -> STR, name of output file containing the plot
    '''
    fig, ax = pca_model.scatter3d()
    outfile = "PCA_3D_scatter_of_{}_inchikeys_occurring_{}_times_in_{}.png"\
        .format(amount, occurrences, pickle_file)
    if os.path.exists(outfile) == False:
        fig.savefig(outfile)
        print("PCA plot stored in file: {}".format(outfile))
    else:
        print("WARNING! File name already exists, new plot will not be made!")
        print("File name: {}".format(outfile))
    return outfile

def perform_tsne(dataset):
    '''Performs t-SNE on a Pandas dataframe with spec2vec vectors.
    
    Input:
        dataset -> Pandas dataframe, contains spec2vec vectors, uses inchikeys
                   as index
    Returns:
        tsne_model -> sklearn.manifold._t_sne.TSNE class
        tsne_results -> numpy array, stored results
    '''
    tsne_model = TSNE(n_components=2, verbose=1)
    tsne_results = tsne_model.fit_transform(dataset)
    return tsne_model, tsne_results

def plot_tsne(dataset, tsne_results, pickle_file, amount, occurrences):
    '''Makes a 2D scatterplot based on t-SNE results and stores it in a .png.
    
    Input:
        dataset -> Pandas dataframe, contains spec2vec vectors, uses inchikeys
                   as index
        tsne_results -> numpy array, stored t-SNE results
        pickle_file -> STR, name of .pickle file containing the spectra
        amount -> INT, amount of inchikeys randomly selected
        occurrences -> INT, min amount of inchikeys that should occur
    Returns:
        outfile -> STR, name of output file containing the plot
    '''
    outfile = "tSNE_2D_scatter_of_{}_inchikeys_occurring_{}_times_in_{}.png"\
        .format(amount, occurrences, pickle_file)
    if os.path.exists(outfile) == False:
        #define x and y axes
        dataset['tsne-2d-one'] = tsne_results[:,0]
        dataset['tsne-2d-two'] = tsne_results[:,1]
        #plot:
        plt.figure(figsize=(20,13)) #adjust plot size
        plot = sns.scatterplot(x = "tsne-2d-one", y = "tsne-2d-two", hue = \
            dataset.index, palette = sns.color_palette("hls", amount),\
            data = dataset, legend = "full")
        figure = plot.get_figure()
        figure.savefig(outfile)
        print("t-SNE plot stored in file: {}".format(outfile))
    else:
        print("WARNING! File name already exists, new plot will not be made!")
        print("File name: {}".format(outfile))
    return outfile

def plot_umap(dataset, umap_results, pickle_file, amount, occurrences):
    '''Makes a 2D scatterplot based on UMAP results and stores it in a .png.
    
    Input:
        dataset -> Pandas dataframe, contains spec2vec vectors, uses inchikeys
                   as index
        umap_results -> numpy array, stored UMAP results
        pickle_file -> STR, name of .pickle file containing the spectra
        amount -> INT, amount of inchikeys randomly selected
        occurrences -> INT, min amount of inchikeys that should occur
    Returns:
        outfile -> STR, name of output file containing the plot
    '''
    outfile = "UMAP_2D_scatter_of_{}_inchikeys_occurring_{}_times_in_{}.png"\
        .format(amount, occurrences, pickle_file)
    if os.path.exists(outfile) == False:
        #define x and y axes
        dataset['UMAP-one'] = umap_results[:,0]
        dataset['UMAP-two'] = umap_results[:,1]
        #plot:
        plt.figure(figsize=(20,13)) #adjust plot size
        plot = sns.scatterplot(x = "UMAP-one", y = "UMAP-two", hue = \
            dataset.index, palette = sns.color_palette("hls", amount),\
            data = dataset, legend = "full")
        figure = plot.get_figure()
        figure.savefig(outfile)
        print("UMAP plot stored in file: {}".format(outfile))
    else:
        print("WARNING! File name already exists, new plot will not be made!")
        print("File name: {}".format(outfile))
    return outfile


if __name__ == "__main__":
    
    #1) Define current working directory and data folder
    ROOT = os.path.dirname(os.getcwd())
    path_data = input("Enter path to data folder: ")
    sys.path.insert(0, ROOT)
    
    #2) Load the pickle file:
    pickle_file = input("Enter name of pickle file to analyze: ")
    spectrums = load_pickle_file(pickle_file, path_data)
    print("{} spectra before filtering.".format(len(spectrums)))
    
    #3) Apply post processing steps to the data
    print("Filtering spectra with matchms..")
    spectrums_s2v = [post_process_s2v(s) for s in spectrums]
    
    #4) Omit spectra that didn't qualify for analysis
    spectrums_s2v = [s for s in spectrums_s2v if s is not None]
    print("{} remaining spectra after filtering.".format(len(spectrums_s2v)))
    
    #5) Make inchikey series in Pandas
    inchikeys_pdseries, inchikey_list = build_inchikey_series(spectrums_s2v)
    
    #6) Build data frame with inchikeys that exist 'n' times in the dataset
    n = int(input("Enter min nr occurences for inchikey (no higher than {}): "\
        .format(inchikeys_pdseries.str[:14].value_counts()[0])))
    suitable_inchikeys = inchikey_selection(n, inchikeys_pdseries)
    print("Number of inchikeys with >={} spectra: {}.".format\
            (n, suitable_inchikeys.shape[0]))
        
    #7) Randomly select a given 'amount' of inchikeys from step 6
    amount = int(input("Enter how many inchikeys to randomly select: "))
    selected_inchikeys = random_selection(suitable_inchikeys, amount)
    
    #8) Find the spectra associated with the selected inchikeys
    selected_spectra = find_spectra(selected_inchikeys, inchikey_list)
    print("The subset contains {} spectra.".format(len(selected_spectra)))
    
    #9) Load pre-trained spec2vec model
    pretrained_model = load_trained_model(input\
        ("Enter filename of pre-trained spec2vec model: "), path_data)
        
    #10) Create spectrum documents
    print("Making spectrum documents..")
    spectrum_documents = [SpectrumDocument(s, n_decimals=2) for i, s in\
        enumerate(spectrums_s2v) if i in selected_spectra]
    
    #11) Calculate vectors and store them in a numpy array
    print("Calculating vectors..")
    vectors = find_vectors(spectrum_documents, pretrained_model)
    
    #12) Make ordered LIST of selected inchikeys to use as index while plotting
    inchikeys_index = make_inchikey_list(spectrums_s2v, selected_spectra)
    
    #13) Create dataset for PCA/tSNE/UMAP
    print("Creating vector data frame..")
    spec_dataset = make_dataset(vectors, inchikeys_index)
    
    #14) Perform PCA
    tstart_pca = time.time()
    print("Performing PCA..")
    pca_model, pca_results = perform_pca(spec_dataset)
    tend_pca = time.time()
    time_pca = tend_pca - tstart_pca
    print("PCA finished in {}h{}m{}s.".format(int(time_pca / 3600), int\
        (time_pca % 3600 / 60), int(time_pca % 3600 % 60)))
    
    #15) Plot PCA results (2D and 3D)
    print("Plotting PCA results in 2D scatter plot..")
    pca_outfile = plot_pca_2D(pca_model, pickle_file, amount, n)
    pca_outfile3d = plot_pca_3D(pca_model, pickle_file, amount, n)
    
    #16) Perform t-SNE
    tstart_tsne = time.time()
    print("Performing t-SNE..")
    tsne_model, tsne_results = perform_tsne(spec_dataset)
    tend_tsne = time.time()
    time_tsne = tend_tsne - tstart_tsne
    print("t-SNE finished in {}h{}m{}s.".format(int(time_tsne / 3600), int\
        (time_tsne % 3600 / 60), int(time_tsne % 3600 % 60)))
    
    #17) Plot t-SNE results (2D)
    print("Plotting t-SNE results in 2D scatter plot..")
    tsne_outfile = plot_tsne(spec_dataset, tsne_results, pickle_file, amount,\
        n)
    
    #18) Perform UMAP
    tstart_umap = time.time()
    print("Performing UMAP..")
    umap_results = umap.UMAP().fit_transform(spec_dataset)
    tend_umap = time.time()
    time_umap = tend_umap - tstart_umap
    print("UMAP finished in {}h{}m{}s.".format(int(time_umap / 3600), int\
        (time_umap % 3600 / 60), int(time_umap % 3600 % 60)))
    
    #19) Plot UMAP results (2D)
    print("Plotting UMAP results in 2D scatter plot..")
    umap_outfile = plot_umap(spec_dataset, umap_results, pickle_file, amount,\
        n)
