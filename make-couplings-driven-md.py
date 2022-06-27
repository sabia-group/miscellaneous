#!/usr/bin/env python3
""" 02.10.2020 Karen Fidanyan
    This script calculates how interatomic distances change
    along the phonon modes provided in xyz_jmol format.

    They can be used in distances formulation of DMD,
    which was implemented in i-PI by Mariana and Karen in March-May 2020.

    One thing that can be improved if needed: one can move
    from absolute to relative distance changes Δdist/dist,
    then it would work great for large systems also.

    It returns a file 'couplings-all.factor_%g.dat' which contains lines
    <frequency of the mode m> <coupling constants for m>
    couplings should be copied to a separate file
    which is then provided to i-pi input.xml.
"""

import sys
import numpy as np
from ase.io import read


def read_xyz_jmol(filename):
    """ Returns list of frames, each frame is a tuple:
        (freq (scalar), nmode (N,3))
    """
    file = open(filename, 'r')
    frames = []
    while(True):
        nmode = []
        # Read the 1st XYZ line with number of atoms
        line = file.readline()
        if not line:
            break
        line2 = line.split()
        nat = int(line2[0])
        # Read .xyz_jmol comment line: "# 0.000020 cm^-1, branch # 1"
        line = file.readline()
        freq = float(line.split()[1])  # Frequency of the mode

        # Read eigenvectors
        for i in range(nat):
            line = file.readline()
            line2 = line.split()
            nmode.append(list(map(float, line2[4:7])))
        nmode = np.array(nmode)

        frames.append((freq, nmode))
    file.close()
    return frames


def vector_separation(cell_h, cell_ih, qi, qj):
    """Calculates the vector separating two atoms.

       Note that minimum image convention is used, so only the image of
       atom j that is the shortest distance from atom i is considered.

       Also note that while this may not work if the simulation
       box is highly skewed from orthorhombic, as
       in this case it is possible to return a distance less than the
       nearest neighbour distance. However, this will not be of
       importance unless the cut-off radius is more than half the
       width of the shortest face-face distance of the simulation box,
       which should never be the case.

       Args:
          cell_h: The simulation box cell vector matrix.
          cell_ih: The inverse of the simulation box cell vector matrix.
          qi: The position vector of atom i.
          qj: The position vectors of one or many atoms j shaped as (N, 3).
       Returns:
          dij: The vectors separating atoms i and {j}.
          rij: The distances between atoms i and {j}.
    """

    sij = np.dot(cell_ih, (qi - qj).T)  # column vectors needed
    sij -= np.rint(sij)

    dij = np.dot(cell_h, sij).T         # back to i-pi shape
    rij = np.linalg.norm(dij, axis=1)

    return dij, rij


if __name__ == "__main__":
    if len(sys.argv) == 4:
        geom_filename = sys.argv[1]  # Name of a geometry file
        vib_filename = sys.argv[2]   # Name of xyz_jmol file with vibrations
        prefac = float(sys.argv[3])  # Prefactor to make coupling small
    else:
        print("3 arguments needed:\n"
              "\t- file with equilibrium geometry "
              "(any ASE-readable format that includes cell)\n"
              "\t  (units angstrom are expected)\n"
              "\t- file with normal modes in xyz_jmol format, "
              "e.g. done by 'eigvec-to-xyz_jmol.py'\n"
              "\t  (normalized normal modes in Cartesian space are expected)\n"
              "\t- prefactor (typically 1e-3)\n"
              "We stop here.")
        sys.exit(-1)

    atoms = read(geom_filename)
    cell = atoms.cell[:]
    invcell = np.linalg.inv(cell)
    modes = read_xyz_jmol(vib_filename)

    orig_dists = []
    for i in range(len(atoms)):
        (_, rij) = vector_separation(cell,
                                     invcell,
                                     atoms.positions[i],
                                     atoms.positions[:i])
        orig_dists += list(rij)
    orig_dists = np.asarray(orig_dists)

    freqs = []
    dlists = []
    for freq, nmode in modes:
        if freq == 'nan':
            freqs.append(0.)
        else:
            freqs.append(freq)
        disp_atoms = atoms.copy()
        disp_atoms.set_positions(atoms.positions + nmode)

        distlist = []
        for i in range(len(atoms)):
            (_, rij) = vector_separation(cell,
                                         invcell,
                                         disp_atoms.positions[i],
                                         disp_atoms.positions[:i])
            distlist += list(rij)

        diff = distlist - orig_dists
        diff = np.around(diff, decimals=6)
        dlists.append(diff)

    dlists = np.asarray(dlists)
    dlists *= prefac      # coupling constant should be small
    data = np.column_stack((freqs, dlists))
    print(data)
    np.savetxt("couplings-all.factor_%g.dat" % prefac,
               data,
               fmt='%g',
               header="Made by make-couplings-dmd.py\n"
                      "Frequency (cm^-1), {constants}")
