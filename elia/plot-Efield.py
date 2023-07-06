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

def main():
    """main routine"""

    # prepare/read input arguments
    print("\tReading script input arguments")
    options = prepare_parser()

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


    Ef = ElectricField( Eamp=data["Eamp"],\
                        Ephase=data["Ephase"],\
                        Efreq=data["Efreq"],\
                        Epeak=data["Epeak"],\
                        Esigma=data["Esigma"])

    # plot of the E-field
    Ef_plot(Ef,data,options)

    # plot of the E-field FFT
    FFT_plot(Ef,data,options)

    # fig, ax1 = plt.subplots(figsize=(12,6))

    # #x = ts["time"]#/4.1341373e3 #np.arange(len(data.occupations[:,0]))/10
    # y = ts["conserved"] #data.energy.sum(axis=1)
    # energy = y - np.mean(y)

    # ax1.plot(x,energy,color="red",label="$\\Delta \\mathcal{H}^{\\mathbf{0}}_n$")
    # ax1.plot(x,np.linspace(energy[0],energy[0],len(x)),color="black",alpha=0.7,linestyle="dashed",label="$\\mathcal{H}^{\\mathbf{0}}_n\\left(t=0\\right)$")

    # ax2 = ax1.twinx()
    # Efield = np.dot(ts["Efield"], v)
    # ax2.plot(x,Efield,color="blue",label="$\\mathbf{E}\\cdot\\mathbf{v}$",alpha=0.4)

    # ax1.set_ylabel("energy (a.u.)")
    # ax2.set_ylabel("E-field (a.u.)")
    # ax1.xaxis.grid()
    # ax1.yaxis.grid()

    # ax1.set_xlabel('time (ps)')
    # ax1.set_title('LiNbO$_3$ (NVE@$0K$,$\\Delta t = 1fs$,E$_{max}$=$10^{-3}$ a.u., 19THz)')
    # ax1.set_xlim(min(x),max(x))

    # fact_dw = 5
    # fact_up = 15

    # n = 7
    # a,b = ax1.get_ylim()
    # c = a - (b-a)*fact_dw/100
    # d = b + (b-a)*fact_up/100
    # ax1.set_yticks(np.linspace(c,d, n))

    # a,b = ax2.get_ylim()
    # c = a - (b-a)*fact_dw/100
    # d = b + (b-a)*fact_up/100
    # ax2.set_yticks(np.linspace(c,d, n))

    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()

    # lines = lines_1 + lines_2
    # labels = labels_1 + labels_2

    # ax1.legend(lines, labels, loc=0)

    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # # Save the figure and show
    # plt.tight_layout()
    # plt.savefig("eda-nve.pdf")


    # #######
    # fig, ax1 = plt.subplots(figsize=(12,6))

    # # x /= 4.1341373e4 # ps

    # N = len(x)
    # dt = x[1]-x[0]
    # T = dt*N
    # df = 1/T
    # dw = 2*np.pi/T
    # ny = dw*N/2
    # w = np.array([dw*n if n<N/2 else dw*(n-N) for n in range(N)])
    # f = np.array([df*n if n<N/2 else df*(n-N) for n in range(N)]) #THz

    # Enfft = np.fft.rfft(energy)
    # freq = np.fft.rfftfreq(len(energy), d=x[1]-x[0])    
    # ax1.plot(freq,np.absolute(Enfft),color="red",label="$\\Delta \\mathcal{H}^{\\mathbf{0}}_n$",alpha=0.4,marker=".")
    # #ax1.plot(w,np.absolute(Enfft),color="green",label="$\\Delta \\mathcal{H}^{\\mathbf{0}}_n$",alpha=0.4,marker=".")
    # #ax1.scatter(freq,np.absolute(Enfft),color="red",marker="x")

    # ax2 = ax1.twinx()
    # Efft  = np.fft.rfft(Efield)
    # freq = np.fft.rfftfreq(len(Efield), d=x[1]-x[0])
    # ax2.plot(freq,np.absolute(Efft),color="blue",label="$\\mathbf{E}\\cdot\\mathbf{v}$",alpha=0.4,marker=".")
    # #ax2.scatter(freq,np.absolute(Efft),color="blue",marker="+",alpha=0.4)

    # ax1.set_yscale("log")
    # ax2.set_yscale("log")
    # ax1.set_xscale("log")

    # ylim = ax1.get_ylim()
    # ax1.set_ylim(1e-4,ylim[1])

    # ax1.yaxis.set_major_locator(mtick.LogLocator(numticks=5))

    # yticks = ax1.get_yticks()
    # ax2.set_yticks(yticks)
    # ax2.yaxis.set_major_locator(mtick.LogLocator(numticks=5))

    # ylim = ax1.get_ylim()
    # ax1.vlines(x=omega/(2*np.pi),ymin=ylim[0],ymax=ylim[1],label="$\\nu$=19THz",color="black",alpha=0.5)
    # #ax1.vlines(x=2*np.pi*19,ymin=ylim[0],ymax=ylim[1],label="$\\omega$=19THz$\\times2\\pi$",color="brown",alpha=0.5)
    # #ax1.vlines(x=19/(2*np.pi),ymin=ylim[0],ymax=ylim[1],label="$\\omega$=19THz$/2\\pi$",color="purple",alpha=0.5)

    # # w = np.loadtxt("nu.txt")
    # # xlim = ax1.get_xlim()
    # # for i in w :
    # #     ax1.vlines(x=i,ymin=ylim[0],ymax=ylim[1],color="black",alpha=0.5)
    # # ax1.set_xlim(*xlim)
    
    # ax1.set_ylabel("energy (a.u.)")
    # ax2.set_ylabel("E-field (a.u.)")
    # ax1.set_xlabel('freq. (THz)')

    # # ax1.set_ylim(1e-4,1e1)
    # #ax2.set_ylim(1e-4,1e1)
    # # n = 7
    # # a,b = ax1.get_ylim()
    # # c = a - (b-a)*fact_dw/100
    # # d = b + (b-a)*fact_up/100

    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()

    # lines = lines_1 + lines_2
    # labels = labels_1 + labels_2

    # ax1.legend(lines, labels, loc=0)
    # ax1.set_title('LiNbO$_3$ (NVE@$0K$,$\\Delta t = 1fs$,E$_{max}$=$10^{-3}$ a.u., 19THz)')

    # #yticks = [ 10**i for i in np.linspace(-10,2,4) ]
    # #ax1.set_yticks(yticks)
    # #ax1.set_ylim(min(yticks)/100,max(yticks))

    # #yticks = [ 10**i for i in np.linspace(-1,1,4) ]
    # #ax2.set_yticks(yticks)
    # #ax2.set_ylim(min(yticks),max(yticks))


    # #plt.xlim(0.001,5)
    # # plt.legend()
    # ax1.grid()#which='both',axis='both')
    # #ax2.grid()
    # plt.tight_layout()
    # plt.savefig("FFT.pdf")



    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()