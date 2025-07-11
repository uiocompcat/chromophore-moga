import os
import sys
import subprocess

def flatten_list(list: list):

    """Flattens a given list.

    Returns:
        list: The flattened list.
    """

    return [item for sublist in list for item in sublist]

with open(sys.argv[1], 'r') as fh:
    xyzs = fh.read().split('\n\n')

names = []
charges = []

for xyz in xyzs:

    comment_line = xyz.split('\n')[1]

    name = comment_line.split(' | ')[0]
    charge = int(comment_line.split('|')[1].replace('charge: ', ''))
    conn = ','.join([str(int(_ + 1)) for _ in flatten_list(eval(comment_line.split('|')[2].replace('conn atom: ', '')))])

    names.append(name)
    charges.append(charge)

    # write molecule to file
    with open('temp_mol.xyz', 'w') as fh:
        fh.write(xyz)

    parameters = [
        '-ligadd temp_mol.xyz',
        '-ligname ' + name,
        '-ligcon ' + conn,
        '-skipANN True'
    ]

    result = subprocess.run(['molsimplify', *parameters], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result)

os.remove('temp_mol.xyz')

