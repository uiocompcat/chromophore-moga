================
Chromophore MOGA
================

This is the code base for a multiobjective genetic algorithm for the *de novo* design of transition metal chromophores. 

Requirements
------------

This package requires a Python (>3.7.x) installation with `molSimplify <https://github.com/hjkgrp/molSimplify>`_.

Furthermore, a working installation of `Gaussian <https://gaussian.com/>`_, used to evaluate the quality of transition metal chromophores, is required.

How to use
----------

The code can be obtained by running::
    
    $ git clone https://github.com/hkneiding/PL-MOGA

which copies the full project into your current working directory.

First, make sure to add all ligands to the ``molSimplify`` ligand library using the ``add_ligands_to_molsimplify.py`` script::

    $ python add_ligands_to_molsimplify.py JT-VAE_selection.xyz
    $ python add_ligands_to_molsimplify.py tmQMg-L_selection.xyz
    
Afterwards, runs can be started directly from the projects root directory using the command line::

    $ python main.py config.yml

The ``config.yml`` file contains entries for all relevant PL-MOGA parameters and is used to configure PL-MOGA runs. 


