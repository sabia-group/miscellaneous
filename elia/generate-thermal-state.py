# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
from classes import MicroState
from functions import str2bool

def prepare_parser():

    parser = argparse.ArgumentParser(description="",\
                                     formatter_class=RawTextHelpFormatter,\
                                        usage=SUPPRESS)

    parser.add_argument(
        "-T", "--temperature", action="store", type=float,metavar='\b',
        help="temperature (Kelvin)"#, default=None
    )
    parser.add_argument(
        "-q", "--relaxed", action="store", type=str,metavar='\b',
        help="input file with the relaxed/original nuclear positons ('xyz' format, a.u.)"#, default=None
    )
    parser.add_argument(
        "-m", "--modes", action="store", type=str,metavar='\b',
        help="folder containing the vibrational modes computed by i-PI"#, default=None
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,metavar='\b',
        help="prefix for the output files", default="start"
    ) 

    parser.add_argument(
        "-f", "--file", action="store", type=str,metavar='\b',
        help="(prefix of the) file containing the state of the random number generator", default="random-state.pickle"
    ) 
    parser.add_argument(
        "-s", "--save", action="store", type=str2bool,metavar='\b',
        help="save the random number generator state", default=True
    ) 
    parser.add_argument(
        "-r", "--read", action="store", type=str2bool,metavar='\b',
        help="read the random number generator state", default=True
    ) 
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    ###
    # prepare/read input arguments
    # print("\n\tReding input arguments")
    options = prepare_parser()

    # read input files
    print("\n\tReding input files for computation")
    data = MicroState(options,what="generate-thermal-state")
    
    print("\n\tGenerating thermal state(s)")
    r,v = data.generate_thermal_state(T=options.temperature,\
                                      randomfile=options.file,\
                                      save=options.save,\
                                      read=options.read)

    file = "{:s}/start_x.xyz".format(options.output)
    print("\n\tSaving positions to file {:s}".format(file))
    data.save2xyz(what=r,file=file)  

    file = "{:s}/start_v.xyz".format(options.output)
    print("\n\tSaving velocities to file {:s}".format(file))
    data.save2xyz(what=v,file=file)    


    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
