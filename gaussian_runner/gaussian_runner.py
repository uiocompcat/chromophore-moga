import os
import warnings
import subprocess
from contextlib import contextmanager
import time
import shutil

from .gaussian_output_parser import GaussianOutputParser

@contextmanager
def change_directory(destination: str):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)

def replace_text(path:str, replacements: dict):
    with open(path, 'r') as file:
        content = file.read()
    for token, value in replacements.items():
        content = content.replace(f'<{token}>', value)
    with open(path, 'w') as file:
        file.write(content)

def check_opt(file_path: str) -> bool: #check if is this a failed optimization?
    """
        Arguments:
            file_path (str): path to the .out of the opt calculation.
        Returns:
            bool: False, if the optimization is successful; True if is indeed a failed opt
    """
    with open(file_path,'r') as f:
        content = f.read()
        if content.count('Normal termination of Gaussian') == 1:
            return False
        else:
            return True

def get_n_step(opt_path: str) -> int: #get the number of the step with the min 'Maximum Force'
    max_force_values = []
    with open(opt_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Maximum Force' in line:
                max_force = line.split()[2]
                max_force_values.append(max_force)
    min_max_force = min(max_force_values)
    n_step = max_force_values.index(min_max_force)
    return n_step + 1

class GaussianRunner:

    """Adapter class for running gaussian jobs of type opt + td-dft."""


    def __init__(self, output_format: str = 'raw'):

        """Constructor.

            Arguments:
                output_format (str): The format to output the result of gaussian calculations.
        """

        # remember the directory from which the program was launched from
        self._root_directory = os.getcwd()

        # validate output format
        supported_output_formats = ['raw', 'dict']
        if output_format not in supported_output_formats:
            warnings.warn('Warning: Output format "' + output_format + '" not recognised. Defaulting to "raw".')
            self._output_format = 'raw'
        else:
            self._output_format = output_format


    def run_gaussian(self, xyz: str):

        """Executes gaussian with the given file.

        Arguments:
            xyz (str): Content of the run.xyz file.
        Returns:
            str: The gaussian output.
        """


        lines = xyz.splitlines()
        n_atoms = int(lines[0])
        list_atoms = []
        coords_list = []
        for i in range(2, n_atoms + 2):
            atom = lines[i].strip().split()[0]
            coords = lines[i]
            coords_list.append(coords)
            if atom != 'Ru':
                if atom not in list_atoms:
                    list_atoms.append(atom)
        non_metal_atoms = ' '.join(str(element) for element in list_atoms)
        xyz = '\n'.join(str(line) for line in coords_list)
        # change to gaussian directory,

        with change_directory(self._root_directory):

            #copy files into folder
            template_run_path = '/templates/run.sh'
            template_com_path = '/templates/individual.com'
            shutil.copy(template_run_path, self._root_directory)
            shutil.copy(template_com_path, self._root_directory)

            #get new file paths

            opt_run_path = os.path.join(self._root_directory, 'run_opt.sh')
            opt_com_path = os.path.join(self._root_directory, 'individual_opt.com')

            path_parts = os.path.normpath(self._root_directory).split(os.sep)

            # Get the generation and individual
            job_name = f'{path_parts[-2]}_{path_parts[-1]}'

            # define replace dic (one for run, one for .com)

            opt_run_replacements = {
                'job_name': job_name,
                'ntasks_per_node': '40',
                'time_limit': '03:30:00',
            }

            opt_gaussian_replacements = {
                'com_name': job_name,
                'xyz': xyz,
                'non_metal_atoms': non_metal_atoms,
            }


            #replace tokens in templates with predefined function
            replace_text(opt_run_path, opt_run_replacements)
            replace_text(opt_com_path, opt_gaussian_replacements)


            # run gaussian optimization
            opt_result = subprocess.run(f'sbatch run.sh individual_opt.com',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            #get job id
            output = opt_result.stdout.decode('utf-8')
            job_id = output.split()[-1]

            # Path to the opt output file
            out_opt_file_path = os.path.join(self._root_directory, 'individual_opt.out')

            # Check opt job completion
            opt_start_time = time.time()
            opt_max_h = 5    # 5-3.5, 1.5h max of queue or
            opt_max_time = opt_max_h*3600  #max time in seconds

            failed_opt = False

            while True:
                opt_elapsed_time = time.time() - opt_start_time
                # Check if the file exists
                if os.path.exists(out_opt_file_path):
                    failed_opt = check_opt(out_opt_file_path) #predefined function, returns False if there is 'normal termination' else True
                    break
                elif opt_elapsed_time > opt_max_time:
                    failed_opt = True
                    break
                time.sleep(60)  # Wait 60 seconds before checking again

            # Get opt chk path
            chk_relative_path = f'$USERWORK/{job_id}'
            chk_absolute_path = os.path.abspath(os.path.expandvars(chk_relative_path))

            if failed_opt == True:
                subprocess.run(f'scancel {job_id}') #if opt fails, kill opt job
                opt_relative_path = f'$USERWORK/{job_id}'  #path to opt in the work folder
                opt_absolute_path = os.path.abspath(os.path.expandvars(opt_relative_path))

                n_step = get_n_step(opt_absolute_path)

                tddft_gaussian_replacements = {
                    'path_chk': chk_absolute_path,
                    'xyz': xyz,
                    'non_metal_atoms': non_metal_atoms,
                    'geom': f'geom=(allcheck,step={n_step}'
                }

            else:
                tddft_gaussian_replacements = {
                    'path_chk': chk_absolute_path,
                    'xyz': xyz,
                    'non_metal_atoms': non_metal_atoms,
                    'geom': f'geom=(allcheck)'
                }

            tddft_run_replacements = {
                'job_name': job_name,
                'ntasks_per_node': '40',
                'time_limit': '01:30:00',
            }


            tddft_run_path = os.path.join(self._root_directory, 'run_tddft.sh')
            tddft_com_path = os.path.join(self._root_directory, 'individual_tddft.com')

            replace_text(tddft_run_path, tddft_run_replacements)
            replace_text(tddft_com_path, tddft_gaussian_replacements)

            #run td-dft
            tddft_max_h = 3
            tddft_max_time = tddft_max_h*3600


            tddft_result = subprocess.run(f'sbatch run.sh individual_tddft.com',
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            # Path to the tddft output file
            out_tddft_file_path = os.path.join(self._root_directory, 'individual_tddft.out')

            tddft_start_time = time.time()
            while True:
                tddft_elapsed_time = time.time() - tddft_start_time
                # Check if the file exists
                if os.path.exists(out_tddft_file_path):
                    with open(out_tddft_file_path, 'r') as file:
                        result_content = file.read()
                    # Return output
                    if self._output_format == 'raw':
                        return result_content
                    elif self._output_format == 'dict':
                        return GaussianOutputParser().parse(result_content)
                elif tddft_elapsed_time > tddft_max_time:
                    raise RuntimeError
                # Sleep for a while before checking again
                time.sleep(60)  # Wait 60 seconds before checking again








