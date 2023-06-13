# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
from classes import MicroState
from functions import vector_type
from ase.io import read
import numpy as np
from functions import output_file,save2xyz,str2bool,print_cell

import argparse

def prepare_parser():

    parser = argparse.ArgumentParser(description="",\
                                     formatter_class=RawTextHelpFormatter,\
                                        usage=SUPPRESS)

    parser.add_argument(
        "-q", "--positions", action="store", type=str,metavar='\b',
        help="file with the positons (xyz, a.u.)"#, default=None
    )
    parser.add_argument(
        "-c", "--cell", action="store", type=str,metavar='\b',
        help="file with the cell parameters (txt, a.u.)"#, default=None
    )
    parser.add_argument(
        "-t", "--transpose", action="store", type=str2bool,metavar='\b',
        help="whether the cell has to be transpose ('true' for Quantum Espresso, 'false' for i-PI)", default=True
    )
    parser.add_argument(
        "-n", "--nx-ny-nz", action="store", type=vector_type,metavar='\b',
        help="super-cell size"#, default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,metavar='\b',
        help="output folder", default="super-cell"
    ) 
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    print("\n\tReding input arguments")
    options = prepare_parser()

    if len(options.nx_ny_nz) != 3:
       raise ValueError("wrong super-cell size")
    options.nx_ny_nz = np.asarray(options.nx_ny_nz)
    if np.any( options.nx_ny_nz <= 0 ):
       raise ValueError("wrong super-cell size")

    print("\tReding positions from file '{:s}'".format(options.positions))
    try :
        tmp = read(options.positions)
        positions = tmp.positions
        names = tmp.get_chemical_symbols()
    except:
        positions = np.loadtxt(options.positions)

    print("\tReding cell parameters from file '{:s}'".format(options.cell))
    try :
        cell = read(options.cell).cell.T
    except:
        cell = np.loadtxt(options.cell)
        if options.transpose :
            cell = cell.T

    string = print_cell(cell)
    print(string)

    print("\tComputing super-cell nuclear positions")

    # Number of Atoms
    Na = len(positions)

    # Super-Cell size
    SC = np.prod(options.nx_ny_nz)

    # Super-Cell Positions
    SCpos = np.full((SC,Na,3),np.nan)
    print("\n\tNumber of atoms in the super-cell: {:>d}".format(SC*Na))

    # subtract 1
    # options.nx_ny_nz -= 1
    nx,ny,nz = options.nx_ny_nz

    # cycle
    k = 0 
    for x in range(nx):
       for y in range(ny):
          for z in range(nz):
            v = cell[:,0]*x + cell[:,1]*y + cell[:,2]*z 
            SCpos[k,:,:] = positions + v
            k += 1

    SCpos = SCpos.reshape((-1,3))

    print("\n\tComputing super-cell lattice parameters")
    SCcell = np.zeros((3,3))
    SCcell[:,0] = cell[:,0] * nx
    SCcell[:,1] = cell[:,1] * nx
    SCcell[:,2] = cell[:,2] * nx

    string = print_cell(SCcell)
    print(string)

    file = output_file(options.output,"positions.xyz")
    print("\tSaving positions tp file '{:s}'".format(file))
    save2xyz(what=SCpos,file=file,atoms=names*SC,comment="super-cell {:d}x{:d}x{:d}".format(nx,ny,nz))
    #write(file,SCpos,format="xyz",fmt="%16.10e")

    if options.transpose :
        SCcell = SCcell.T

    file = output_file(options.output,"cell.txt")
    print("\tSaving positions tp file '{:s}'".format(file))
    np.savetxt(file,SCcell,fmt="%16.10e")

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

