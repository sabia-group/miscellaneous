# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

# this fiel contains some useful functions

import argparse
import os
import itertools
import numpy as np
import re

# https://stackabuse.com/python-how-to-flatten-list-of-lists/
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def get_all_system_permutations(atoms):
    species   = np.unique(atoms)
    index     = {key: list(np.where(atoms == key)[0]) for key in species}
    # permutations = {key: get_all_permutations(i) for i,key in zip(index.values(),species)}
    permutations = [get_all_permutations(i) for i in index.values()]
    return list(itertools.product(*permutations))

def get_all_permutations(v):
    tmp = itertools.permutations(list(v))
    return [ list(i) for i in tmp ]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_one_file_in_folder(folder,ext):
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            files.append(os.path.join(folder, file))
    if len(files) == 0 :
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1 :
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

def get_property_header(inputfile,N=1000):

    names = [None]*N
    restart = False

    with open(inputfile, "r") as ifile:
        icol = 0        
        while True:
            line = ifile.readline()
            nline = line
            if not line:
                break
            elif "#" in line:
                line = line.split("-->")[1]
                line = line.split(":")[0]
                line = line.split(" ")[1]

                nline = nline.split("-->")[0]
                if "column" in nline:
                    lenght = 1
                else :
                    nline = nline.split("cols.")[1]
                    nline = nline.split("-")
                    a,b = int(nline[0]),int(nline[1])
                    lenght = b - a  + 1 

                if icol < N :
                    if lenght == 1 :
                        names[icol] = line
                        icol += 1
                    else :
                        for i in range(lenght):
                            names[icol] = line + "-" + str(i)
                            icol += 1
                else :
                    restart = True
                    icol += 1
                
            
    if restart :
        return get_property_header(inputfile,N=icol)
    else :
        return names[:icol]

def getproperty(inputfile, propertyname,data=None,skip="0"):

    def check(p,l):
        if not l.find(p) :
            return False # not found
        elif l[l.find(p)-1] != " ":
            return False # composite word
        elif l[l.find(p)+len(p)] == "{":
            return True
        elif l[l.find(p)+len(p)] != " " :
            return False # composite word
        else :
            return True

    if type(propertyname) in [list,np.ndarray]: 
        out   = dict()
        units = dict()
        data = np.loadtxt(inputfile)
        for p in propertyname:
            out[p],units[p] = getproperty(inputfile,p,data,skip=skip)
        return out,units
    
    print("\tsearching for '{:s}'".format(propertyname))

    skip = int(skip)

    # propertyname = " " + propertyname + " "

    # opens & parses the input file
    with open(inputfile, "r") as ifile:
        # ifile = open(inputfile, "r")

        # now reads the file one frame at a time, and outputs only the required column(s)
        icol = 0
        while True:
            try:
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
                while "#" in line :  # fast forward if line is a comment
                    line = line.split(":")[0]
                    if check(propertyname,line):
                        cols = [ int(i)-1 for i in re.findall(r"\d+", line) ]                    
                        if len(cols) == 1 :
                            icol += 1
                            output = data[:,cols[0]]
                        elif len(cols) == 2 :
                            icol += 1
                            output = data[:,cols[0]:cols[1]+1]
                        elif len(cols) != 0 :
                            raise ValueError("wrong string")
                        if icol > 1 :
                            raise ValueError("Multiple instances for '{:s}' have been found".format(propertyname))

                        l = line
                        p = propertyname
                        if l[l.find(p)+len(p)] == "{":
                            unit = l.split("{")[1].split("}")[0]
                        else :
                            unit = "atomic_unit"

                    # get new line
                    line = ifile.readline()
                    if len(line) == 0:
                        raise EOFError
                if icol <= 0:
                    print("Could not find " + propertyname + " in file " + inputfile)
                    raise EOFError
                else :
                    return np.asarray(output),unit

            except EOFError:
                break


def vector_type(arg_value):
    try:
        # Split the input string by commas and convert each element to an integer
        values = [int(x) for x in arg_value.split(',')]
        return values
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid vector: {arg_value}") from e
    
def output_folder(folder):
    if folder in ["",".","./"] :
        folder = "."
    elif not os.path.exists(folder) :
        print("\n\tCreating directory '{:s}'".format(folder))
        os.mkdir(folder)
    return folder

def output_file(folder,what):
    folder = output_folder(folder)
    return "{:s}/{:s}".format(folder,what)

def save2xyz(what,file,atoms,comment=""):

    if len(what.shape) == 1 : # just one configuration, NOT correctly formatted

        what = what.reshape((-1,3))
        return save2xyz(what,file,atoms)
    
    elif len(what.shape) == 2 : 

        if what.shape[1] != 3 : # many configurations
            what = what.reshape((len(what),-1,3))
            return save2xyz(what,file,atoms)
        
        else : # just one configurations, correctly formatted
            return save2xyz(np.asarray([what]),file,atoms)

    elif len(what.shape) == 3 :

        Na = what.shape[1]
        if what.shape[2] != 3 :
            raise ValueError("wrong shape")
        
        with open(file,"w") as f :
            
            for i in range(what.shape[0]):
                pos = what[i,:,:]
                f.write(str(Na)+"\n")
                f.write("# {:s}\n".format(comment))
                for ii in range(Na):
                    f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))
        return
    

def print_cell(cell,tab="\t\t"):
    cell = cell.T
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string