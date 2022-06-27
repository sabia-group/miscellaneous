#!/usr/bin/python3
""" This script takes geometry file (in any ASE-supported format)
    and performs Phono3py analysis: builds displacements and
    after they are calculated (outside of this script),
    builds the 3rd order force constant matrix
    and calculates phonon self-energy (gamma).

    Naturally, Phono3py stores data in hdf5 format,
    but here gammas and eigenfrequencies at each q-point are extracted as a postpocessing.

    Adjustable parameters in the beginning of 'main' section:
    - infile = 'geometry.unitcell.in' - initial structure to analyze
    - supercell_matrix=np.diag([4, 4, 1])
    - mesh = [21, 21, 1] - mesh of q-points to output gammas and frequencies
    - temperatures=[80]  - list of temperatures to calculate gammas
    - symprec=1e-5       - symmetry precision. Not recommended to change it

    - cutoff_pair_distance = 6.0 - cutoff dist. for phonon-phonon interaction
        Displacements involving atoms beyond cutoff will be excluded.
        It is recommended to start with lower values
        and increase it until convergence.
        The displacements are enumerated in such a way
        that adding further ones keeps previous valid,
        therefore one need only to calculate additional points.

    For a detailed description of output files see
    https://atztogo.github.io/phono3py/output-files.html
"""

import os
import sys
import numpy as np
from ase.atoms import Atoms
from ase.io import read, write
from phono3py import Phono3py


def to_phonopy_atoms(structure, wrap=False):
    """ (c) Florian Knoop, Hilde project (a.k.a. FHI-vibes)
    Converts ase.atoms.Atoms to PhonopyAtoms
    Parameters:
    structure: ase.atoms.Atoms  - Atoms to convert
    wrap:      bool             - If True wrap the scaled positions

    Returns
    phonopy_atoms: PhonopyAtoms - The PhonopyAtoms for the same structure
    """
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_atoms = PhonopyAtoms(
        symbols=structure.get_chemical_symbols(),
        cell=structure.get_cell(),
        masses=structure.get_masses(),
        positions=structure.get_positions(wrap=wrap),
    )
    return phonopy_atoms


def to_Atoms(structure, info=None, pbc=True):
    """ (c) Florian Knoop, Hilde project (a.k.a. FHI-vibes)
    Convert structure to ase.atoms.Atoms
    Parameters:
    structure: PhonopyAtoms  - The structure to convert
    info:      dict          - Additional information to include in atoms.info
    pbc:       bool          - True if the structure is periodic

    Returns
    atoms:     ase.atoms.Atoms  - The ASE representation of the material
    """

    if info is None:
        info = {}
    if structure is None:
        return None

    atoms_dict = {
        "symbols": structure.get_chemical_symbols(),
        "cell": structure.get_cell(),
        "masses": structure.get_masses(),
        "positions": structure.get_positions(),
        "pbc": pbc,
        "info": info,
    }

    atoms = Atoms(**atoms_dict)
    return atoms


def create_supercells_with_displacements(phono3py, cutoff_pair_distance=None):
    phono3py.generate_displacements(distance=0.03,
                                    cutoff_pair_distance=cutoff_pair_distance
                                    )
    scells_with_disps = phono3py.get_supercells_with_displacements()
    print("%i displacements exist." % len(scells_with_disps))

    excluded = 0
    for i, scell in enumerate(scells_with_disps):
        if scell is None:
            # print("phono3py-displacement-%05d is excluded." % (i+1))
            excluded += 1
            continue
        # write("displacements_all.xyz", to_Atoms(scell), append=True)
        if not os.path.isdir("phono3py-displacement-%05d" % (i+1)):
            os.mkdir("phono3py-displacement-%05d" % (i+1))
        # I don't want to rewrite all geometries each time I run the script
        if os.path.exists("phono3py-displacement-%05d/geometry.in" % (i+1)):
            if to_Atoms(scell) == read("phono3py-displacement-%05d/geometry.in"
                                       % (i+1)):
                print("phono3py-displacement-%05d/geometry.in "
                      "exists and is valid."
                      % (i+1))
                continue
        write("phono3py-displacement-%05d/geometry.in"
              % (i+1),
              to_Atoms(scell))

    print("%i displacements are included."
          % (len(scells_with_disps) - excluded))
    disp_dataset = phono3py.get_displacement_dataset()
    print("Duplucates:")
#    print(disp_dataset['duplicates'])
    print("(not printed now)")

    # Print displacements to screen
    print("Set of displacements:")
    print("(not printed now)")
    count = 0
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        # print("%4d: %4d                %s" % (
        #     count + 1,
        #     disp1['number'] + 1,
        #     np.around(disp1['displacement'], decimals=3)))
        count += 1
    distances = []
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        for j, disp2 in enumerate(disp1['second_atoms']):
            # print("%4d: %4d-%4d (%6.3f)  %s %s \tincluded: %s" % (
            #     count + 1,
            #     disp1['number'] + 1,
            #     disp2['number'] + 1,
            #     disp2['pair_distance'],
            #     np.around(disp1['displacement'], decimals=3),
            #     np.around(disp2['displacement'], decimals=3),
            #     disp2['included']))
            distances.append(disp2['pair_distance'])
            count += 1
    # Find unique pair distances
    distances = np.array(distances)
    unique_thres = 1e-5  # Threshold of identity for finding unique distances
    distances_int = (distances / unique_thres).astype(int)
    unique_distances = np.unique(distances_int) * unique_thres
    print("Unique pair distances")
    print(unique_distances)


# ==================== HERE THE ALGORITHM STARTS ====================
if __name__ == '__main__':
    infile = 'geometry.unitcell.in'
    supercell_matrix = np.diag([4, 4, 1])
    mesh = [21, 21, 1]
    temperatures = [80]
    symprec = 1e-5
    cutoff_pair_distance = 6.0
    print("cutoff_pair_distance: %.2f" % cutoff_pair_distance)

    atoms = read(infile, format='aims')
    print("Cell:")
    print(atoms.get_cell()[:])
    cell = to_phonopy_atoms(atoms, wrap=False)
    phono3py = Phono3py(unitcell=cell,
                        supercell_matrix=supercell_matrix,
                        mesh=mesh,
                        symprec=symprec,
                        log_level=2)  # log_level=0 make phono3py quiet

    print("Cell symmetry by international table:  %s"
          % phono3py._symmetry._international_table)
    print("Supercell symmetry by international table:  %s"
          % phono3py._phonon_supercell_symmetry._international_table)
    print("Symmetry precision %g" % symprec)
    print("Primitive cell:")
    print(phono3py._primitive)

#    sys.exit(0)

    create_supercells_with_displacements(
        phono3py,
        cutoff_pair_distance=cutoff_pair_distance
        )

    # ============== Postprocessing after forces are calculated ==============
#    print("phono3py._supercell.get_atomic_numbers():",
#          phono3py._supercell.get_atomic_numbers())
    zero_force = np.zeros([len(phono3py._supercell.get_atomic_numbers()), 3])

    # collect the forces and put zeros where no supercell was created
    force_sets = []
    disp_scells = phono3py.get_supercells_with_displacements()
#    print(phono3py._displacement_dataset)
    for nn, scell in enumerate(disp_scells):
        if not scell:
            force_sets.append(zero_force)
        else:
            try:
                with open("phono3py-displacement-%05d/aims.out"
                          % (nn+1), 'r') as f:
                    lines = f.read().splitlines()
                    if "nice day" not in lines[-2]:
                        print("ERROR: phono3py-displacement-%05d/aims.out "
                              "doesn't have 'nice day'. Leaving now."
                              % (nn+1))
                        sys.exit(-1)
                atoms = read("phono3py-displacement-%05d/aims.out" % (nn+1))
            except:
                print("Cannot read 'phono3py-displacement-%05d/aims.out'\n"
                      "Make sure that all displacements are calculated."
                      % (nn+1))
                sys.exit(-1)
            force_sets.append(atoms.get_forces())

#    print("force_sets:")
#    print(force_sets)
    phono3py.produce_fc3(force_sets)
    # How grid_points are treated:
    # ~/.local/lib/python3.5/site-packages/phono3py/phonon3/conductivity.py
    phono3py.run_thermal_conductivity(temperatures=temperatures,
                                      boundary_mfp=1e6,  # in micrometre
                                      # solve_collective_phonon=False,
                                      # use_ave_pp=False,
                                      # gamma_unit_conversion=None,
                                      # gv_delta_q=None,  # for group velocity
                                      write_gamma=True,
                                      write_kappa=True,
                                      write_gamma_detail=True,
                                      write_collision=True,
                                      # write_pp=True,
                                      )

# --------------- ~/soft/phono3py/example/Si-PBEsol/Si.py --------------
#             Has some useful pieces, also it's a reference.
# ----------------------------------------------------------------------
