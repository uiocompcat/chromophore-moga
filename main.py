import os
import sys
import yaml
import pickle
import functools
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from gaussian_runner.gaussian_runner import GaussianRunner


element_identifiers = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                        'Bi', 'Po', 'At', 'Rn']

transition_metal_atomic_numbers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,                           # first block
                                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,                          # second block
                                    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,      # lanthanides
                                    72, 73, 74, 75, 76, 77, 78, 79, 80,                              # third block
                                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                                    104, 105, 106, 107, 108, 109, 110, 111, 112]                     # fourth block

@contextmanager
def change_directory(destination: str):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)



def compose(*functions):

    """Composes a given list of functions.

    Returns:
        callable: The composed function.
    """

    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)

def flatten_list(list: list):

    """Flattens a given list.

    Returns:
        list: The flattened list.
    """

    return [item for sublist in list for item in sublist]

def charge_range(individual, charges, allowed_charges):

    """Checks if a given individual is within the allowed charge range.

    Returns:
        bool: The flag indicating whether the individual is within the allowed charge range.
    """

    if calculate_total_charge(individual, charges) in allowed_charges:
        return True
    
    return False

def determine_electron_count(individual):

    """Determines the electron count of a molecule individual.

    Returns:
        int: The electron count.
    """

    xyz_lines = individual.meta['initial_xyz'].strip().split('\n')[2:]

    electron_count = -individual.meta['oxidation_state']
    for xyz_line in xyz_lines:
        element = xyz_line.split()[0]
        electron_count += element_identifiers.index(element) + 1

    return electron_count

def is_singlet(individual):

    """Checks if a given individual is singlet.

    Returns:
        bool: The flag indicating if individual is singlet.
    """

    if determine_electron_count(individual) % 2 == 0:
        return True

    return False

def are_oct_equivalents(l1: list, l2: list):

    """Checks if two genomes of bidentate ligands correspond to the same TMC.

    Arguments:
        l1 (list): The first genome.
        l2 (list): The second genome.

    Returns:
        bool: The flag indicating whether or not the genomes correspond to the same TMC.

    Raises:
        AssertionError: If two ligands with the same id do not have the same is_symmetric property.
    """

    l1_ids = [_['id'] for _ in l1]
    l1_is_symmetric = [_['is_symmetric'] for _ in l1]
    l1_is_flipped = [_['is_flipped'] for _ in l1]

    l2_ids = [_['id'] for _ in l2]
    l2_is_symmetric = [_['is_symmetric'] for _ in l2]
    l2_is_flipped = [_['is_flipped'] for _ in l2]

    # check for erroneous ligand data
    ids = l1_ids + l2_ids
    is_symmetric = l1_is_symmetric + l2_is_symmetric
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            # assert that any two ligands with the same id also have the same
            # is_symmetric property
            if ids[i] == ids[j]:
                assert is_symmetric[i] == is_symmetric[j]

    # check if different ligands are used
    if set(l1_ids) != set(l2_ids):
        return False

    # ABABAB pattern
    if len(list(set(l1_ids))) == 1 and sum(l1_is_symmetric) == 0:

        if sum(l1_is_flipped) == sum(l2_is_flipped):
            return True
        if sum(l1_is_flipped) == 0 and sum(l2_is_flipped) == 3:
            return True
        if sum(l1_is_flipped) == 3 and sum(l2_is_flipped) == 0:
            return True
        if sum(l1_is_flipped) == 1 and sum(l2_is_flipped) == 2:
            return True
        if sum(l1_is_flipped) == 2 and sum(l2_is_flipped) == 1:
            return True        

    # AAAAAA pattern
    elif len(list(set(l1_ids))) == 1 and sum(l1_is_symmetric) == 3:

        return True
    
    # AAAABB pattern
    elif len(list(set(l1_ids))) == 2 and sum(l1_is_symmetric) == 3:

        return True

    # ABABCC pattern
    elif len(list(set(l1_ids))) == 2 and sum(l1_is_symmetric) == 1:
        
        l1_extended = l1 * 2
        for i, _ in enumerate(l1_extended):
            if l1_extended[i]['is_symmetric'] == 0 and l1_extended[i + 1]['is_symmetric'] == 0:
                normalized_l1 = l1_extended[i:i+3]
                break

        l2_extended = l2 * 2
        for i, _ in enumerate(l2_extended):
            if l2_extended[i]['is_symmetric'] == 0 and l2_extended[i + 1]['is_symmetric'] == 0:
                normalized_l2 = l2_extended[i:i+3]
                break

        if (normalized_l1[0]['is_flipped'] == normalized_l2[0]['is_flipped']) and \
           (normalized_l1[1]['is_flipped'] == normalized_l2[1]['is_flipped']):
            return True
        
        if (normalized_l1[0]['is_flipped'] == 1) and \
           (normalized_l1[1]['is_flipped'] == 1) and \
           (normalized_l2[0]['is_flipped'] == 0) and \
           (normalized_l2[1]['is_flipped'] == 0):
            return True 

        if (normalized_l1[0]['is_flipped'] == 0) and \
           (normalized_l1[1]['is_flipped'] == 0) and \
           (normalized_l2[0]['is_flipped'] == 1) and \
           (normalized_l2[1]['is_flipped'] == 1):
            return True 

    # ABCCCC pattern
    elif len(list(set(l1_ids))) == 2 and sum(l1_is_symmetric) == 2:
        
        return True

    # ABABCD pattern
    elif len(list(set(l1_ids))) == 2 and sum(l1_is_symmetric) == 0:

        l1_extended = l1 * 2
        for i, _ in enumerate(l1_extended):
            if l1_extended[i]['id'] == l1_extended[i + 1]['id']:
                normalized_l1 = l1_extended[i:i+3]
                break

        l2_extended = l2 * 2
        for i, _ in enumerate(l2_extended):
            if l2_extended[i]['id'] == l2_extended[i + 1]['id']:
                normalized_l2 = l2_extended[i:i+3]
                break

        if (normalized_l1[0]['is_flipped'] == normalized_l2[0]['is_flipped']) and \
           (normalized_l1[1]['is_flipped'] == normalized_l2[1]['is_flipped']) and \
           (normalized_l1[2]['is_flipped'] == normalized_l2[2]['is_flipped']):
            return True

        if (normalized_l1[0]['is_flipped'] == normalized_l1[1]['is_flipped']) and \
           (normalized_l2[0]['is_flipped'] == normalized_l2[1]['is_flipped']) and \
           (normalized_l1[0]['is_flipped'] != normalized_l2[0]['is_flipped']) and \
           (normalized_l1[1]['is_flipped'] != normalized_l2[1]['is_flipped']) and \
           (normalized_l1[2]['is_flipped'] != normalized_l2[2]['is_flipped']):
            return True

        if (normalized_l1[0]['is_flipped'] == 0) and \
           (normalized_l1[1]['is_flipped'] == 0) and \
           (normalized_l1[2]['is_flipped'] == 0) and \
           (normalized_l2[0]['is_flipped'] == 1) and \
           (normalized_l2[1]['is_flipped'] == 1) and \
           (normalized_l2[2]['is_flipped'] == 1):
            return True 

        if (normalized_l1[0]['is_flipped'] == 1) and \
           (normalized_l1[1]['is_flipped'] == 1) and \
           (normalized_l1[2]['is_flipped'] == 1) and \
           (normalized_l2[0]['is_flipped'] == 0) and \
           (normalized_l2[1]['is_flipped'] == 0) and \
           (normalized_l2[2]['is_flipped'] == 0):
            return True 

        if (normalized_l1[0]['is_flipped'] == 0) and \
           (normalized_l1[1]['is_flipped'] == 1) and \
           (normalized_l1[2]['is_flipped'] == 0) and \
           (normalized_l2[0]['is_flipped'] == 0) and \
           (normalized_l2[1]['is_flipped'] == 1) and \
           (normalized_l2[2]['is_flipped'] == 1):
            return True 

        if (normalized_l1[0]['is_flipped'] == 1) and \
           (normalized_l1[1]['is_flipped'] == 0) and \
           (normalized_l1[2]['is_flipped'] == 0) and \
           (normalized_l2[0]['is_flipped'] == 1) and \
           (normalized_l2[1]['is_flipped'] == 0) and \
           (normalized_l2[2]['is_flipped'] == 1):
            return True 

    # ABCDEF pattern
    elif len(list(set(l1_ids))) == 3 and sum(l1_is_symmetric) == 0:
        
        normalized_l1 = l1[0:]

        l2_extended = l2 * 2
        for i, _ in enumerate(l2_extended):
            if normalized_l1[0]['id'] == l2_extended[i]['id']:

                if normalized_l1[1]['id'] == l2_extended[i+1]['id']:
                    normalized_l2 = l2_extended[i:i+3]
                    l2_is_rotated = False
                else:
                    normalized_l2 = [l2_extended[i], l2_extended[i-1], l2_extended[i-2]]
                    l2_is_rotated = True

                break

        if l2_is_rotated:
            
            if (normalized_l1[0]['is_flipped'] != normalized_l2[0]['is_flipped']) and \
               (normalized_l1[1]['is_flipped'] != normalized_l2[1]['is_flipped']) and \
               (normalized_l1[2]['is_flipped'] != normalized_l2[2]['is_flipped']):
                return True 
            
        else:
            
            if (normalized_l1[0]['is_flipped'] == normalized_l2[0]['is_flipped']) and \
               (normalized_l1[1]['is_flipped'] == normalized_l2[1]['is_flipped']) and \
               (normalized_l1[2]['is_flipped'] == normalized_l2[2]['is_flipped']):
                return True 


    # AABBCC pattern
    elif len(list(set(l1_ids))) == 3 and sum(l1_is_symmetric) == 3:

        return True

    # AABBCD pattern
    elif len(list(set(l1_ids))) == 3 and sum(l1_is_symmetric) == 2 :

        l1_extended = l1 * 2
        for i, _ in enumerate(l1_extended):
            if l1_extended[i]['is_symmetric'] == 1 and l1_extended[i+1]['is_symmetric'] == 1:
                normalized_l1 = l1_extended[i:i+3]
                break

        l2_extended = l2 * 2
        for i, _ in enumerate(l2_extended):
            if normalized_l1[0]['id'] == l2_extended[i]['id']:

                if normalized_l1[1]['id'] == l2_extended[i+1]['id']:
                    normalized_l2 = l2_extended[i:i+3]
                    l2_is_rotated = False
                else:
                    normalized_l2 = [l2_extended[i], l2_extended[i-1], l2_extended[i-2]]
                    l2_is_rotated = True
                break

        if l2_is_rotated:

            if (normalized_l1[2]['is_flipped'] != normalized_l2[2]['is_flipped']):
                return True 
            
        else:
            
            if (normalized_l1[2]['is_flipped'] == normalized_l2[2]['is_flipped']):
                return True 

    # AABCDE pattern
    elif len(list(set(l1_ids))) == 3 and sum(l1_is_symmetric) == 1:

        l1_extended = l1 * 2
        for i, _ in enumerate(l1_extended):
            if l1_extended[i]['is_symmetric'] == 1:
                normalized_l1 = l1_extended[i:i+3]
                break

        l2_extended = l2 * 2
        for i, _ in enumerate(l2_extended):
            if normalized_l1[0]['id'] == l2_extended[i]['id']:

                if normalized_l1[1]['id'] == l2_extended[i+1]['id']:
                    normalized_l2 = l2_extended[i:i+3]
                    l2_is_rotated = False
                else:
                    normalized_l2 = [l2_extended[i], l2_extended[i-1], l2_extended[i-2]]
                    l2_is_rotated = True
                break

        if l2_is_rotated:
            
            if (normalized_l1[1]['is_flipped'] != normalized_l2[1]['is_flipped']) and \
               (normalized_l1[2]['is_flipped'] != normalized_l2[2]['is_flipped']):
                return True 
            
        else:
            
            if (normalized_l1[1]['is_flipped'] == normalized_l2[1]['is_flipped']) and \
               (normalized_l1[2]['is_flipped'] == normalized_l2[2]['is_flipped']):
                return True 

    return False



def are_rotation_equivalents(l1: list, l2: list):

    """Checks if two lists are rotational equivalents of each other.

    Arguments:
        l1 (list): The first list.
        l2 (list): The second list.

    Returns:
        bool: The flag indicating whether or not the lists are rotation equivalent.
    """

    if len(l1) != len(l2):
        return False

    l1_extended = l1 * 2
    for i in range(len(l1_extended) - len(l1)):
        if l1_extended[i:i + len(l1)] == l2:
            return True
    
    return False

def zero_mask_target_by_population_median(individual, individuals, target_indices, scaling=1):

    """Masks one fitness target of the individual by zeroing below median of population.

    Arguments:
        individual (Individual): The current individual.
        individuals (list[Individual]: The list of individuals.
        target_indices (list[int]): The target indices to zero mask.
        scaling: (list[float]): The scalings to apply to each of the target indices.

    Returns:
        list: The masked fitness
    """

    for target_idx in target_indices:
      
        target_population = []
        for _ in individuals:
            target_population.append(_._fitness[target_idx])
        
        if individual._fitness[target_idx] < scaling[target_idx] * np.median(target_population):
            return [0, 0]

    return individual._fitness

def parse_xyz(xyz: str):

    """Parses a given xyz into two lists for atoms and positions respectively.

    Returns:
        list[str]: The list of atom identifiers.
        list[list[float]]: The list of atomic positions.
    """

    atoms = []
    positions = []

    lines = xyz.split('\n')
    for i in range(2, len(lines), 1):

        line_split = lines[i].split()

        if len(line_split) != 4:
            break

        atoms.append(line_split[0])
        positions.append([float(line_split[i]) for i in [1, 2, 3]])

    return atoms, positions

def get_metal_connecting_indices(xyz: str, radius_cutoff: float):

    """Gets the indices of atoms connecting to metal based on a cutoff value.

    Returns:
        list[int]: The list of indices connecting to metal.
    """

    atoms, positions = parse_xyz(xyz)

    metal_indices = []
    for i, atom in enumerate(atoms):
        if (element_identifiers.index(atom) + 1) in transition_metal_atomic_numbers:
            metal_indices.append(i)

    connecting_indices = []
    for metal_index in metal_indices:

        for i, position in enumerate(positions):

            if metal_index == i:
                continue

            distance = np.linalg.norm(np.array(position) - np.array(positions[metal_index]))
            if distance <= radius_cutoff:
                connecting_indices.append(i)

    return connecting_indices

def get_radius_adjacency_matrix(positions: list, radius_cutoff: float):

    """Gets an adjacency matrix based on a radius cutoff.

    Returns:
        list[list[int]]: The adjacency matrix.
    """

    adjacency_matrix = np.zeros((len(positions), len(positions)))

    for i in range(len(positions)):
        for j in range(i+1, len(positions), 1):
            
            if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) <= radius_cutoff:
                adjacency_matrix[i,j] = 1
                adjacency_matrix[j,i] = 1

    return adjacency_matrix

def radius_graph_is_connected(positions: list, radius_cutoff: float):

    """Checks if a radius graph based on a cutoff is connected or not.

    Returns:
        bool: The flag saying whether the graph is connected or not.
    """

    adj = get_radius_adjacency_matrix(positions, radius_cutoff)
    n_connected_components = connected_components(adj)[0]

    if n_connected_components == 1:
        return True
    return False

def get_electron_count_from_xyz(xyz: str, charge: int = 0):

    """Get the number of electrons from a xyz file.

    Returns:
        int: The electron count.
    """

    lines = xyz.split('\n')

    n_electrons = 0
    for line in lines:

        line_split = line.split()
        if len(line_split) == 4:
            n_electrons += element_identifiers.index(line_split[0]) + 1

    return n_electrons - charge

def calculate_total_charge(individual, charges: list):

    """Calculates the total charge of an individual based on the metal oxidation state
    and the charges of ligands.

    Returns:
        int: The total charge.
    """

    return individual.meta['oxidation_state'] + sum([int(charges[_]) for _ in individual.genome])

def calculate_penalized_logp(individual):

    """Calculates the penalized logP as suggest by 
    GÓMEZ-BOMBARELLI, et al., ACS central science, 2018 (https://arxiv.org/abs/1610.02415v1).

    Returns:
        float: The penalized logP score.
    """

    return sum([gene['penalized_logP'] for gene in individual.genome])

def calculate_sum_logP(individual):

    """Calculates the sum of logP values of a given individual (TMC).

    Returns:
        float: The sum of logP values.
    """

    sum_logp = 0
    for gene in individual.genome:

        sum_logp += gene['logP']

    return sum_logp

def calculate_solubility(dipole_moment: float, sum_logp: float):
        
    """Calculates the solubility based on the dipole moment and sum of logP values.

    Returns:
        float: The calculated solubility
    """

    return dipole_moment/(np.exp(sum_logp))

def fitness_function(individual):

    """Calculates the fitness of a organometallic compound using xTB based on quantum properties.

    Returns:
        float: The fitness.
    """

    # make unique run directory
    tmp_dir = 'fitness_calculations/' + individual.meta['id']
    os.makedirs(tmp_dir)

    with change_directory(tmp_dir):

        # write individual to file
        with open('individual.txt', 'w') as fh:
            fh.write(str(individual.as_dict()))

        # assemble molsimplify ligand names with the correct flipping
        molsimplify_ligand_names = []
        for gene in individual[genome]:
            if gene[is_flipped]:
                molsimplify_ligand_names.append(gene['id'] + '_flipped')
            else:
                molsimplify_ligand_names.append(gene['id'])

        # prepare molsimplify parameters. Unchanged, maybe we want to fix de core, oxstate etc etc. ?
        parameters = ['-skipANN True',
                      '-core ' + individual.meta['metal_centre'],
                      '-geometry ' + individual.meta['coordination_geometry'],
                      '-lig ' + ','.join([_ for _ in molsimplify_ligand_names]),
                      #'-coord ' + str(len(individual.genome)),
                      '-ligocc ' + ','.join(['1' for _ in individual.genome]),
                      '-name run',
                      '-oxstate ' + str(individual.meta['oxidation_state'])
                      ]

        # run molsimplify
        result = subprocess.run(['molsimplify', *parameters], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #####################################
        ########### TO BE REMOVED ###########
        #####################################

        # read the xyz from molsimplify output
        with open('./Runs/run/run/run.xyz') as fh:
            xyz = fh.read()
            individual.meta['initial_xyz'] = xyz

        fitness_vector = [np.random.rand(), np.random.rand(), np.random.rand()]
        individual.set_fitness(fitness_vector)
        return individual

        #####################################
        ########### TO BE REMOVED ###########
        #####################################

        try:
            # read the xyz from molsimplify output
            with open('./Runs/run/run/run.xyz') as fh:
                xyz = fh.read()
                individual.meta['initial_xyz'] = xyz

            gaussian_runner = GaussianRunner(output_format='dict')

            gaussian_result_dict = gaussian_runner.run_gaussian(xyz)
            individual.meta['optimised_xyz'] = gaussian_result_dict['optimised_xyz']

            sum_logp = calculate_sum_logP(individual)
            dipole_moment = result['dipole_moment']
            solubility = calculate_solubility(dipole_moment, sum_logp)

        except FileNotFoundError:
            print('molSimplify failure. Genome: ' + ','.join([gene['id'] for gene in individual.genome]))
            individual.set_fitness([0, 0])
            return individual

        except RuntimeError:
            print('Gaussian failure. Genome: ' + ','.join([gene['id'] for gene in individual.genome]))
            individual.set_fitness([0, 0])
            return individual

        except Exception:
            print('Other error')
            print(xyz)
            individual.set_fitness([0, 0])
            return individual

    # check that connecting atoms are the same between initial and optimised xyz
    initial_connecting_indices = get_metal_connecting_indices(individual.meta['initial_xyz'], 2.5)
    optimised_connecting_indices = get_metal_connecting_indices(individual.meta['optimised_xyz'], 2.5)
    if initial_connecting_indices != optimised_connecting_indices:
        print('Connecting indices different')
        individual.set_fitness([0, 0])
        return individual

    # check for disconnected graphs
    is_connected = radius_graph_is_connected(parse_xyz(individual.meta['optimised_xyz'])[1], 3.0)
    if not is_connected:
        print('Graph not connected')
        individual.set_fitness([0, 0])
        return individual

    # update fitness and return
    fitness_vector = [
        solubility,
        gaussian_result_dict['n*deltaL'],
        gaussian_result_dict['f_max']
    ]
    individual.set_fitness(fitness_vector)

    return individual

def parse_config_file(file_path: str):

    """Parses a config file and returns a dictionary with all relevant parameters.

    Returns:
        dict: The parameter dictionary.
    """

    with open(file_path, 'r') as fh:
        config = yaml.safe_load(fh)

    #TODO to be moved to config file
    config['ligand_library'] = pd.read_csv('tmQMg-L_selection.csv').to_dict('records')

    # add is_flipped flag
    for _ in config['ligand_library']:
        _['is_flipped'] = False

    # specify ligand space and charges
    if config['parent_selection'] == 'roulette_wheel_rank':
        config['parent_selection'] = roulette_wheel_rank
    elif config['parent_selection'] == 'select_by_rank':
        config['parent_selection'] = select_by_rank
    else:
        print('Config not found:', config['parent_selection'])

    if config['parent_rank'] == 'rank_dominate':
        config['parent_rank'] = rank_dominate
    elif config['parent_rank'] == 'rank_is_dominated':
        config['parent_rank'] = rank_is_dominated
    elif config['parent_rank'] == 'rank_non_dominated_fronts':
        config['parent_rank'] = rank_non_dominated_fronts
    else:
        print('Config not found:', config['parent_rank'])

    if config['survivor_selection'] == 'roulette_wheel_rank':
        config['survivor_selection'] = roulette_wheel_rank
    elif config['survivor_selection'] == 'select_by_rank':
        config['survivor_selection'] = select_by_rank
    else:
        print('Config not found:', config['survivor_selection'])

    if config['survivor_rank'] == 'rank_dominate':
        config['survivor_rank'] = rank_dominate
    elif config['survivor_rank'] == 'rank_is_dominated':
        config['survivor_rank'] = rank_is_dominated
    elif config['survivor_rank'] == 'rank_non_dominated_fronts':
        config['survivor_rank'] = rank_non_dominated_fronts
    else:
        print('Config not found:', config['survivor_rank'])

    if config['crossover'] == 'uniform_crossover':
        config['crossover'] = uniform_crossover
    else:
        print('Config not found:', config['crossover'])

    for i, _ in enumerate(config['mutations']):
        if config['mutations'][i] == 'swap_mutation':
            config['mutations'][i] = functools.partial(
                swap_mutation,
                mutation_rate=config['mutation_rates'][i])
        elif config['mutations'][i] == 'uniform_integer_mutation':
            config['mutations'][i] = functools.partial(
                uniform_integer_mutation, 
                mutation_space=config['ligand_library'], 
                mutation_rate=config['mutation_rates'][i]
            )
        elif config['mutations'][i] == 'flip_mutation':
            config['mutations'][i] = functools.partial(
                flip_mutation, 
                mutation_rate=config['mutation_rates'][i]
            )
        else:
            print('Config not found:', config['mutations'][i])

    # compose mutations
    config['composed_mutation'] = config['mutations'][0]
    for i in range(1, len(config['mutations']), 1):
        config['composed_mutation'] = compose(config['composed_mutation'], config['mutations'][i])

    return config
    
def get_existing_ligand_names():
    
    """Return the ligand names present in the molSimplify library.

    Returns:
        list: The list of existing ligand names.
    """

    out = str(
        subprocess.run(
            ["molsimplify", *["-h", "liganddict"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    )

    return [_.strip() for _ in out.replace("}", "").split("\\n")[8:-1]]

if __name__ == "__main__":

    from gapy.individual import Individual
    from gapy.population import Population
    from gapy.mutation import uniform_integer_mutation, swap_mutation, flip_mutation
    from gapy.crossover import uniform_crossover
    from gapy.selection import select_by_rank, roulette_wheel_rank
    from gapy.rank import rank_dominate, rank_is_dominated, rank_non_dominated_fronts
    from gapy.ga import GA

    config = parse_config_file(sys.argv[1])
    
    # check that each ligand in the library is present in the molSimplify local library
    existing_ligands = get_existing_ligand_names()
    for _ in config['ligand_library']:
        if _['id'] not in existing_ligands:
            print('Ligand ' + _['id'] + ' not in molSimplify library. Exiting..')
            exit()

    # fix random seed
    np.random.seed(config['seed'])

    print('Using ' + str(len(config['ligand_library'])) + ' ligands.')

    # set up GA
    ga = GA(fitness_function=functools.partial(fitness_function),
            parent_selection=functools.partial(config['parent_selection'], n_selected=config['n_parents'], rank_function=config['parent_rank']),
            survivor_selection=functools.partial(config['survivor_selection'], n_selected=config['n_population'], rank_function=config['survivor_rank']),
            crossover=functools.partial(config['crossover'], mixing_ratio=config['crossover_mixing']),
            mutation=config['composed_mutation'],
            n_offspring=config['n_offspring'],
            n_allowed_duplicates=config['n_allowed_duplicates'],
            solution_constraints=[functools.partial(is_singlet)],
            genome_equivalence_function=are_rotation_equivalents,
            masking_function=functools.partial(zero_mask_target_by_population_median, target_indices=[i for i in range(len(config['zeromask_scaling_factors']))], scaling=config['zeromask_scaling_factors'])
    )

    # random initial population
    initial_individuals = []
    for i in range(config['n_population']):
        genome = np.random.choice(config['ligand_library'], size=3).tolist()
        initial_individuals.append(Individual(genome=genome, meta={
            'metal_centre': 'Ru',
            'oxidation_state': 2,
            'coordination_geometry': 'oct',
            'id': 'gen0/ind' + str(i)
        }))
    initial_population = Population(initial_individuals)
    
    # run ga
    final_pop, log = ga.run(n_generations=config['n_generations'], initial_population=initial_population)

    # save log
    with open('log.pickle', 'wb') as fh:
        pickle.dump(log, fh)
