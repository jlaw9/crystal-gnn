import os
import gzip
import json
import yaml
import itertools
import copy


def load_config_file(config_file):
    print(f"Loading config file '{config_file}'")
    with open(config_file, 'r') as f:
        # self.config_map = yaml.safe_load(f)
        # expandvars is a neat trick to expand bash variables within the yaml file
        # from here: https://stackoverflow.com/a/60283894/7483950
        config_map = yaml.safe_load(os.path.expandvars(f.read()))
    return config_map


def load_structures_from_json(structures_file):
    # TODO I get an error when importing pymatgen if I use the wrong package versions
    from pymatgen.core.structure import Structure

    print(f"Loading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())

    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)

    print(f"\t{len(structures)} loaded")
    return structures


def get_experiments(experiments):
    """
    For a given experiment, create copies 
        based on the lists of evaluations
        such that all combinations of settings are run

    The example experiment would be copied 5 times, once for each seed
        - datasets: ["icsd", "battery_relaxed"]
          eval_settings:
              icsd: [random_subset: 0.05]
              hypo: ["leave_out_comp"]
              # random seed to use for train/valid/test splits
              seed: [1,2,3,4,5]
    """
    exp_combos = []
    for exp in experiments:
        eval_settings = exp['eval_settings']
        combos = [dict(zip(eval_settings, val))
            for val in itertools.product(
                *(eval_settings[key] for key in eval_settings))]

        for combo in combos:
            new_exp = copy.deepcopy(exp)
            new_exp['eval_settings'] = combo
            exp_combos.append(new_exp)
    return exp_combos 


eval_names = {
    'random_subset': 'randsub',
    'leave_out_comp': 'loc',
    'leave_out_comp_minus_one': 'loc1',
}

def get_eval_str(config_map, experiment):
    eval_settings = experiment['eval_settings']
    dataset_types = set()
    for dataset_name in experiment['datasets']:
        dataset = config_map['datasets'][dataset_name]
        if dataset['hypothetical'] is True:
            dataset_types.add('hypo')
        else:
            dataset_types.add('icsd')

    eval_strs = []
    for dataset_type in sorted(dataset_types):
        lattice = eval_settings.get(dataset_type + '_lattice', 'orig')
        lattice_str = get_lattice_str(lattice)

        eval_type = eval_settings.get(dataset_type, {'random_subset': 0.05})
        if isinstance(eval_type, dict):
            eval_type, val = list(eval_type.items())[0]
        if eval_type == 'random_subset':
            eval_type = eval_names[eval_type] + str(val).replace('.', '_')
        eval_str = f"{dataset_type}_{lattice_str}{eval_names.get(eval_type, eval_type)}"
        eval_strs.append(eval_str)
    full_eval_str = "_".join(eval_strs)

    rand_seed = eval_settings.get("seed")
    full_eval_str += "_seed" + str(rand_seed) if rand_seed is not None else ""
    return full_eval_str


#struc.lattice = struc.lattice.cubic(1.0)                  
#struc.lattice = struc.lattice.tetragonal(1.0,3.0)         
#struc.lattice = struc.lattice.orthorhombic(1.0,2.0,3.0) 
lattice_nicknames = {
        "cubic": 'c',  # cubic lattice with a=b=c=1
        'monoclinic': 'm',
        "tetragonal": 't',  # tetragonal lattice with a=b=1,c=3
        "orthorhombic": 'o',  # orthorhombic lattice with a=1,b=2,c=3
        }

def get_lattice_str(lattice):
    # if this is the original lattice, just use an empty string
    if lattice == 'orig':
        return ""
    lattice_str = lattice_nicknames.get(lattice,lattice) + 'latt_'
    return lattice_str


def get_out_dir(config_map, experiment):
    # structure of outputs:
    # <base_output_dir>/
    #   <dataset_name>/
    #     <evaluation>_<seed>/
    #       train/valid/test splits
    #       <hyperparameters>/
    #         trained model
    base_output_dir = config_map['output_dir'] 
    dataset_names = '_'.join(experiment['datasets'])
    eval_str = get_eval_str(config_map, experiment)

    out_dir = os.path.join(base_output_dir, dataset_names, eval_str)
    return out_dir


def get_hyperparam_dir(out_dir, params):
    # now get the directory in which to put this model file (distinguished by hyperparameters)
    model_dir = f"lr{params['learning_rate']:0.0e}_ep{params['epochs']}_bs{params['batch_size']}" + \
                f"_ms{params['max_sites']}_mb{params['max_bonds']}_ed{params['embed_dimension']}" + \
                f"_nm{params['num_messages']}"
    model_dir = f"{out_dir}/{model_dir}"
    return model_dir


def check_default_hyperparams(params):
    params['batch_size'] = int(params.get('batch_size', 64))   # typical batch sizes= 32, 64, 128, 256 (usually batch size=32 or 64 is optimum) 
    params['max_sites'] = int(params.get('max_sites', 256))
    params['max_bonds'] = int(params.get('max_bonds', 2048))
    params['embed_dimension'] = int(params.get('embed_dimension', 256))   # typical embedding dimensions= 32, 64, 128, 256
    params['num_messages'] = int(params.get('num_messages', 6))        # typical number of message passing blocks= 3-6
    params['learning_rate'] = float(params.get('learning_rate', 1E-4))       # too high learning rates can lead to NaN losses 
    params['epochs'] = int(params.get('epochs', 100))
    return params


