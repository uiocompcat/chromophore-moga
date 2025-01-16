

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


class GaussianOutputParser:
    """Class for parsing gaussian td-dft output."""

    def __init__(self):

        """Constructor."""

        self.lines = None
        self.n_atoms = None
    @staticmethod
    def parse(data: str):

        """Parses a given gaussian td-dft output line by line to extract different properties.

        Arguments:
            data (str): The gaussian output.

        Returns:
            dict: A dictionary containing the different outputs.
        """

        if data.count('Normal termination of Gaussian') != 1:
            raise RuntimeError
        else:

            nms, fs, f_max, nm_f_max, dipole_moment, opt_xyz = parse_gaussian_tddft_spectrum(data)

            delta_l, nm_max, nm_min = get_delta_l(nms)

            n_transitions = int(len(fs))

            n_x_delta_l = get_spread(delta_l, n_transitions)

            results = {
                'f_max': f_max,
                'n*deltaL': n_x_delta_l,
                'nm_min': nm_min,
                'nm_max': nm_max,
                'deltaL': delta_l,
                'n_transitions': n_transitions,
                'dipole_moment': float(dipole_moment),
                'optimized_xyz' : opt_xyz
            }

            return results


def parse_gaussian_tddft_spectrum(data: str):

    lines = data.splitlines()
    opt_completed_line = float('inf')
    nms = []
    fs = []
    dipole_moments = []
    optimised_xyz_lines = []
    n_atoms_found = False
    counter = 0
    for line in lines:
        counter += 1
        if not n_atoms_found:
            if 'NAtoms=' in line:
                n_atoms = int(line.split()[1])
                n_atoms_found = True
        if 'Excited State' in line:
            f = float(line.split()[8].split('=')[-1])
            nm = float(line.split()[6])
            if f >= 0.05 and 350 <= nm <= 825:
                nms.append(nm)
                fs.append(f)
        elif 'Dipole moment (field-independent basis, Debye):' in line:
            dipole_line = counter
            dipole = lines[dipole_line].split()[-1]
            dipole_moments.append(dipole)
        elif 'Center     Atomic      Atomic' in line:
            start_index = counter + 2


    for i in range(start_index, start_index + n_atoms):
        xyz_line = []
        for j in (1, 3, 4, 5):
            xyz_line.append(lines[i].split()[j])
        optimised_xyz_lines.append(' '.join(xyz_line))

    opt_xyz = str(len(optimised_xyz_lines)) + '\n\n' + '\n'.join(
        optimised_xyz_lines)  # same format as xtb output parser



    dipole_moment = float(dipole_moments[-1])
    try:
        f_max = max(fs)
        f_max_index = fs.index(f_max)
        nm_f_max = nms[f_max_index]
    except ValueError:
        f_max = 0
        nm_f_max = 0
    return nms, fs, f_max, nm_f_max, dipole_moment, opt_xyz


def get_delta_l(nms: list):
    if len(nms) != 0:
        nm_min = min(nms)
        nm_max = max(nms)
        delta_l = nm_max - nm_min
        return delta_l, nm_max, nm_min
    else:
        return 0, 0, 0

def get_spread(delta_l: float, n_transitions: int):
    if delta_l != 0 and n_transitions != 0:
        return delta_l * n_transitions
    else:
        return 0







