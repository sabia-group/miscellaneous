#!/usr/bin/env python3
from sys import argv
from ase.io import read, write

inname = argv[1]
outname = argv[2]

atoms = read(inname, format="aims")
write(outname, atoms, format="xyz")
