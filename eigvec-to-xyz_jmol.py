#!/usr/bin/env python3
""" 04.05.2020 Karen Fidanyan
    This script takes the relaxed geometry in XYZ format
    and the .mode, .eigval files produced by i-PI,
    and builds a .xyz_jmol file to visualize vibrations.
"""

import sys
import numpy as np
from ase.io import read

if len(sys.argv) == 5:
    xyzname = sys.argv[1]
    vecname = sys.argv[2]
    valname = sys.argv[3]
    outname = sys.argv[4]
else:
    print("4 arguments needed:\n"
          "\t- .xyz file (typically init.xyz)\n"
          "\t  (units angstrom are expected)\n"
          "\t- file with phonon displacement vectors "
          "(typically phon.phonons.mode)\n"
          "\t  (atomic units are expected)\n"
          "\t- file with eigenvalues (typically phon.phonons.eigval)\n"
          "\t  (atomic units are expected)\n"
          "\t- output filename\n"
          "We stop here.")
    sys.exit(-1)

atoms = read(xyzname, format='xyz')
vecmat = np.loadtxt(vecname)
freqs = np.loadtxt(valname)
freqs = np.sqrt(freqs) * 219474.63  # atomic frequency to invcm

np.set_printoptions(formatter={'float': '{: .8f}'.format})
with open(outname, 'w') as fdout:
    for b, vec in enumerate(vecmat.T):
        disp = vec.reshape(-1, 3)
        fdout.write("%i\n# %f cm^-1, branch # %i\n"
                    % (len(atoms), freqs[b], b))
        for i, atom in enumerate(atoms.positions):
            fdout.write("%s  " % atoms[i].symbol
                        + ' '.join(map("{:10.8g}".format, atom)) + "  "
                        + ' '.join(map("{:12.8g}".format, disp[i])) + "\n")
