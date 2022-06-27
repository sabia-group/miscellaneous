#!/usr/bin/env python3
from sys import argv
from ase.io import read

from ipi.engine.atoms import Atoms as Atoms_ipi
from ipi.engine.cell import Cell as Cell_ipi
from ipi.utils.io.backends.io_xyz import print_xyz

if len(argv) == 3:
    inname = argv[1]
    outname = argv[2]
else:
    print("Error: 2 arguments expected.\n\tinname\n\toutname")
    exit(1)

atoms_ase = read(inname, format="aims")

# Careful: here I don't do any unit conversions and use raw io_xyz
# which just spits out what it receives without any conversions.
atoms_ipi = Atoms_ipi(len(atoms_ase))
atoms_ipi.q = atoms_ase.get_positions().flatten()
atoms_ipi.names = atoms_ase.symbols

# We need to transpose the cell because i-PI and ASE use different conventions
cell_ipi = Cell_ipi(atoms_ase.cell.T)

with open(outname, 'w') as fdout:
    print_xyz(atoms_ipi,
              cell_ipi,
              fdout,
              title="positions{angstrom}  cell{angstrom}")

print("Done.")
