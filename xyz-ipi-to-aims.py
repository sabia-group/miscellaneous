#!/usr/bin/env python3
info="""This script converts XYZ file with i-PI cell to aims geometry format.
Attention: works only with XYZ files IN ANGSTROMS (because I'm lazy)!
Arguments:
    inname - a XYZ file with the cell in i-PI format.
    outname - name for Aims geometry file.
"""

from sys import argv
import numpy as np
from ase.io import read, write
from ipi.utils.io import read_file_raw


if len(argv) == 3:
    inname = argv[1]
    outname = argv[2]
else:
    print("Error: wrong arguments.")
    print(info)
    exit(1)

with open(inname) as infile:
    # Read with i-pi reader first, extract the cell and check if it's constant
    rr = read_file_raw('xyz', infile)
    metainfo = rr['comment'].split()
    if ("positions{angstrom}" not in metainfo) or ("cell{angstrom}" not in metainfo):
        raise RuntimeError("Only input in Angstroms is supported so far.")
    cell = rr['cell']

cell = cell.T # i-PI and ASE use different transpositions of the cell matrix.
cell = np.around(cell, decimals=8)

atoms = read(inname, format='xyz')
atoms.pbc = True
atoms.cell = cell
write(outname, atoms, format="aims", info_str="Created by xyz-ipi-to-aims.py")

print("Done.")
