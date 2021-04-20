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
from typing import List, Union
from rdkit import Chem
from matchms.typing import SpectrumType


def read_pickled_spectra(input_file: str) -> List[SpectrumType]:
    """Read pickled spectra

    :param input_file: file path of pickled spectra
    :return: list of spectra
    """
    if os.path.exists(input_file):
        print("\nReading spectra")
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
    except (urllib.error.HTTPError, urllib.error.URLError):
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
    inchikey = ""
    if smiles:
        m = Chem.MolFromSmiles(smiles)
        if m:
            new_smiles = Chem.MolToSmiles(m, kekuleSmiles=False,
                                          isomericSmiles=False)
            if new_smiles:
                new_m = Chem.MolFromSmiles(new_smiles)
                if new_m:
                    inchikey = Chem.inchi.MolToInchiKey(new_m)
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


def get_cf_classes(smiles: str, inchi: str) -> Union[None, List[str]]:
    """Get ClassyFire classes through GNPS API

    :param smiles: Smiles for the query spectrum
    :param inchi: Inchikey for the query spectrum
    :return: ClassyFire classes if possible
    """
    result = None
    # lookup CF with smiles
    if smiles:
        url_base = "https://gnps-structure.ucsd.edu/classyfire?smiles="
        url_smiles = url_base + smiles
        smiles_result = do_url_request(url_smiles)

        # read CF result
        if smiles_result is not None:
            result = get_json_cf_results(smiles_result)

    if not result:
        # do a second try with inchikey
        if inchi:
            url_inchi = \
                f"https://gnps-classyfire.ucsd.edu/entities/{inchi}.json"
            inchi_result = do_url_request(url_inchi)

            # read CF result from inchikey lookup
            if inchi_result is not None:
                result = get_json_cf_results(inchi_result)
    return result


def get_npc_classes(smiles: str) -> Union[None, List[str]]:
    """Get NPClassifier classes through GNPS API

    :param smiles: Smiles for the query spectrum
    :return: NPClassifier classes if possible
    """
    result = None
    # lookup NPClassifier with smiles
    if smiles:
        url_base_npc = "https://npclassifier.ucsd.edu/classify?smiles="
        url_smiles_npc = url_base_npc + smiles
        smiles_result_npc = do_url_request(url_smiles_npc)

        # read NPC result
        if smiles_result_npc is not None:
            result = get_json_npc_results(smiles_result_npc)
    return result


def get_classes(spectra: List[SpectrumType]) -> List[List[str]]:
    """Get classes for the spectra via GNPS API

    :param spectra: list of spectra
    :return: list of list of the required info per spectrum
    """
    print("\nRetrieving classes from GNPS API")
    classes = []
    missed_cfs = 0
    missed_npcs = 0
    for i, spec in enumerate(spectra):
        if i % 5000 == 0 and not i == 0:
            print(f"{i} spectra done")

        # get info for spectrum
        spec_id = spec.metadata.get("spectrum_id")
        if not spec_id:  # as a check if it will have id under different name
            spec_id = spec.metadata.get("spectrumid")
        smiles = spec.metadata.get("smiles")
        if not smiles:
            smiles = ""  # smiles can be None in metadata
            print(f"\t#{i} {spec_id} no smiles")
        inchi = spec.metadata.get("inchikey")
        # try to retrieve inchikey from smiles through rdkit
        if not inchi:
            inchi = inchikey_from_smiles_rdkit(smiles)
            if inchi:
                print(f"\t#{i} {spec_id} retrieved inchikey with smiles")
            else:
                print(f"\t#{i} {spec_id} no inchikey")

        smiles = smiles.strip(' ')
        safe_smiles = urllib.parse.quote(smiles)  # url encoding
        cf_result = get_cf_classes(safe_smiles, inchi)
        if not cf_result:
            missed_cfs += 1
            # num classes we want, if they are changed, change this number
            cf_result = ['' for _ in range(5)]

        npc_result = get_npc_classes(safe_smiles)
        if not npc_result:
            # num classes we want, if they are changed, change this number
            npc_result = ['' for _ in range(4)]
        # superclass, im assuming this one occurs the most if missing others
        if not npc_result[1]:
            missed_npcs += 1

        # combine results
        combined_result = [str(i), spec_id, smiles, inchi] + \
            cf_result + npc_result
        classes.append(combined_result)

    print(f"Retrieved {len(classes) - missed_cfs} ClassyFire classes," +
          f" missing {missed_cfs}")
    print(f"Retrieved {len(classes) - missed_npcs} NPClassifier " +
          f"superclasses, missing {missed_npcs}")
    return classes


def write_class_info(classes: List[List[str]], out_file: str):
    """Write classes to out_file

    :param classes: list of list of the required info per spectrum
    :param out_file: location of output file
    """
    if not out_file.endswith('.txt'):
        out_file += '.txt'
    print("\nWriting output to:", out_file)

    header_list = [
        'spectrum_index', 'spectrum_id', 'smiles', 'inchi_key', 'cf_kingdom',
        'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent',
        'npc_class_results', 'npc_superclass_results', 'npc_pathway_results',
        'npc_isglycoside']
    with open(out_file, 'w') as outf:
        outf.write("{}\n".format('\t'.join(header_list)))
        for classes_per_spec in classes:
            assert len(classes_per_spec) == len(header_list)
            outf.write("{}\n".format('\t'.join(classes_per_spec)))


if __name__ == "__main__":
    tstart = time.time()
    error_msg = "Incorrect input" + \
                f"\nUsage:\n\tpython {argv[0]} <spectra.pickle> <output_file>"
    if len(argv) != 3:
        raise ValueError(error_msg)
    if not argv[1].endswith('.pickle'):
        raise ValueError(error_msg)
    print("\nStart")

    spectrums = read_pickled_spectra(argv[1])
    classes_result = get_classes(spectrums)
    write_class_info(classes_result, argv[2])

    tend = time.time()
    t = tend - tstart
    t_str = '{}h{}m{}s'.format(int(t / 3600), int(t % 3600 / 60),
                               int(t % 3600 % 60))
    print('\nFinished in {}'.format(t_str))
