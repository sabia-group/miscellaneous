#!/usr/bin/env python3

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

# python setup.py build
# python setup.py install

import argparse
import matplotlib.pyplot as plt
import numpy as np
from ipi.engine.ensembles import ElectricField
import xml.etree.ElementTree as xmlet
import os
#import ast 
from functions import convert

def plt_clean():
    ###
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def compute(Ef,data,options):
    factor = convert ( 1 , "time" , options.unit , "picosecond" )
    t = np.arange(0,options.t_max,options.time_spacing) * factor
    tt = t * convert ( 1 , "time" , "picosecond" , "atomic_unit" )
    E = np.zeros( (len(t),3))    
    E = Ef._get_Efield(tt)
    f = Ef._get_Eenvelope(tt) * np.linalg.norm(data["Eamp"])
    En = np.linalg.norm(E,axis=1)
    return t,E,En,f

def FFT_plot(Ef,data,options):
    t,E,En,f= compute(Ef,data,options)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(t,f,label="$f_{env} \\times E_{amp}$",color="black")
    ax.plot(t,En,label="$|E|$",color="gray",alpha=0.5)
    ax.plot(t,E[:,0],label="$E_x$",color="red",alpha=0.5)
    ax.plot(t,E[:,1],label="$E_y$",color="green",alpha=0.5)
    ax.plot(t,E[:,2],label="$E_z$",color="blue",alpha=0.5)

    plt.ylabel("electric field [a.u.]")
    plt.xlabel("time [ps]")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    temp = os.path.splitext(options.output) 
    file = "{:s}.FFT{:s}".format(temp[0],temp[1])
    print("\tSaving plot to {:s}".format(file))
    plt.savefig(file)

def Ef_plot(Ef,data,options):

    t,E,En,f= compute(Ef,data,options)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t,f,label="$f_{env} \\times E_{amp}$",color="black")
    ax.plot(t,En,label="$|E|$",color="gray",alpha=0.5)
    ax.plot(t,E[:,0],label="$E_x$",color="red",alpha=0.5)
    ax.plot(t,E[:,1],label="$E_y$",color="green",alpha=0.5)
    ax.plot(t,E[:,2],label="$E_z$",color="blue",alpha=0.5)

    plt.ylabel("electric field [a.u.]")
    plt.xlabel("time [ps]")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    print("\tSaving plot to {:s}".format(options.output))
    plt.savefig(options.output)

    plt_clean()

def prepare_parser():
    """set up the script input parameters"""

    parser = argparse.ArgumentParser(description="Plot the electric field E(t) into a pdf file.")

    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file", default="input.xml"
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file ", default="Efield.pdf"
    )
    parser.add_argument(
        "-t", "--t-max", action="store", type=float,
        help="max time",
    )
    parser.add_argument(
        "-dt", "--time-spacing", action="store", type=float,
        help="max time",default=1
    )
    parser.add_argument(
        "-u", "--unit", action="store", type=str,
        help="unit",default="picosecond"
    )
       
    options = parser.parse_args()

    return options

def get_data(options):

    print("\tReading json file")
    # # Open the JSON file and load the data
    # with open(options.input) as f:
    #     info = json.load(f)

    data = xmlet.parse(options.input).getroot()

    ensemble = None
    for element in data.iter():
        if element.tag == "ensemble":
            ensemble = element
            break

    data     = {}
    keys     = ["Eamp",          "Efreq",    "Ephase",   "Epeak","Esigma"]
    families = ["electric-field","frequency","undefined","time", "time"  ]
    
    for key,family in zip(keys,families):

        data[key] = None
        
        element = ensemble.find(key)

        if element is not None:
            #value = ast.literal_eval(element.text)
            text =  element.text
            try :
                value = text.split('[')[1].split(']')[0].split(',')
                value = [ float(i) for i in value ]
                if len(value) == 1:
                    value = float(value)
                else :
                    value = np.asarray(value)
            except :
                value = float(text)
            
            try :
                unit = element.attrib["units"]
                if unit is None :
                    unit = "atomic_unit"
            except:
                unit = "atomic_unit"

            # print(key,value,unit)

            value = convert(value,family,unit,"atomic_unit")
            data[key] = value

    return data

def main():
    """main routine"""

    # prepare/read input arguments
    print("\tReading script input arguments")
    options = prepare_parser()

    data = get_data(options)

    Ef = ElectricField( Eamp=data["Eamp"],\
                        Ephase=data["Ephase"],\
                        Efreq=data["Efreq"],\
                        Epeak=data["Epeak"],\
                        Esigma=data["Esigma"])

    # plot of the E-field
    Ef_plot(Ef,data,options)

    # plot of the E-field FFT
    #FFT_plot(Ef,data,options)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()