#!/usr/bin/env python
"""
Script to find ClassyFire and NPClassifier classes for matchms.SpectrumType via
GNPS API.
Makes a file with:
index spectrum_id smiles inchikey cf_classes npc_classes
"""

import os
import json
import pickle
import urllib
import time
from sys import argv
from typing import List
from rdkit import Chem
from matchms.typing import SpectrumType


def read_pickled_spectra(input_file: str) -> List[SpectrumType]:
    """Read pickled spectra

    :param input_file: file path of pickled spectra
    :return: list of spectra
    """
    if os.path.exists(input_file):
        with open(input_file, 'rb') as inf:
            spectra = pickle.load(inf)
    else:
        raise FileNotFoundError(f"{input_file} does not exist")
    return spectra


def do_url_request(url: str) -> [bytes, None]:
    """
    Do url request and return bytes from .read() or None if HTTPError is raised

    :param url: url to access
    :return: open file or None if request failed
    """
    try:
        with urllib.request.urlopen(url) as inf:
            result = inf.read()
    except urllib.error.HTTPError:
        # apparently the request failed
        result = None
    return result


def get_json_cf_results(raw_json: bytes) -> List[str]:
    """
    Extract the wanted CF classes from bytes version (open file) of json str

    Names of the keys extracted in order are:
    'kingdom', 'superclass', 'class', 'subclass', 'direct_parent'
    List elements are concatonated with '; '.

    :param raw_json: Json str as a bytes object containing ClassyFire
        information
    :return: Extracted CF classes
    """
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list_name = ['kingdom', 'superclass', 'class',
                             'subclass', 'direct_parent']
    for key in wanted_keys_list_name:
        info_dict = cf_json.get(key, "")
        info = ""
        if info_dict:
            info = info_dict.get('name', "")
        wanted_info.append(info)

    return wanted_info


def inchikey_from_smiles_rdkit(smiles: str) -> str:
    """Use rdkit to go from smiles to inchikey

    :param smiles: Smiles to be turned into inchikey
    :return: inchikey as string
    """
    m = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(m, kekuleSmiles=False, isomericSmiles=False)
    m = Chem.MolFromSmiles(smiles)
    inchikey = Chem.inchi.MolToInchiKey(m)
    return inchikey


def get_json_npc_results(raw_json: bytes) -> List[str]:
    """Read bytes version of json str, extract the keys in order

    Names of the keys extracted in order are:
    class_results, superclass_results, pathway_results, isglycoside.
    List elements are concatonated with '; '.

    :param raw_json:Json str as a bytes object containing NPClassifier
        information
    :return: Extracted NPClassifier classes
    """
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list = ["class_results", "superclass_results",
                        "pathway_results"]
    # this one returns a bool not a list like the others
    last_key = "isglycoside"

    for key in wanted_keys_list:
        info_list = cf_json.get(key, "")
        info = ""
        if info_list:
            info = "; ".join(info_list)
        wanted_info.append(info)

    last_info_bool = cf_json.get(last_key, "")
    last_info = "0"
    if last_info_bool:
        last_info = "1"
    wanted_info.append(last_info)

    return wanted_info


if __name__ == "__main__":
    tstart = time.time()
    error_msg = "Incorrect input" +\
        f"\nUsage:\n\tpython {argv[0]} <spectra.pickle> <output_file>"
    if len(argv) != 3:
        raise ValueError(error_msg)
    if not argv[1].endswith('.pickle'):
        raise ValueError(error_msg)
    print("\nStart")

    spectrums = read_pickled_spectra(argv[1])

    tend = time.time()
    t = tend-tstart
    t_str = '{}h{}m{}s'.format(int(t / 3600), int(t % 3600 / 60),
                               int(t % 3600 % 60))
    print('\nFinished in {}'.format(t_str))
