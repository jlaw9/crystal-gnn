import os
import gzip
import json
import yaml

from pymatgen.core.structure import Structure


def load_config_file(config_file):
    print(f"Loading config file '{config_file}'")
    with open(config_file, 'r') as f:
        # self.config_map = yaml.safe_load(f)
        # expandvars is a neat trick to expand bash variables within the yaml file
        # from here: https://stackoverflow.com/a/60283894/7483950
        config_map = yaml.safe_load(os.path.expandvars(f.read()))
    return config_map


def load_structures_from_json(structures_file):
    print(f"Loading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())

    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)

    print(f"\t{len(structures)} loaded")
    return structures

