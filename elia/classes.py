from functions import get_one_file_in_folder,getproperty,output_file,get_property_header
from functions import convert, Dict2Obj, get_attributes, merge_attributes, read_comments_xyz
import os
from ase import io #.io import read
from ase import Atoms
import numpy as np
import numpy.linalg as linalg
norm = linalg.norm
import matplotlib.pyplot as plt
from ipi.utils.units import unit_to_internal, unit_to_user
import pickle
import numpy.random as rand
import pandas as pd
from reloading import reloading
import re
import ipi.utils.mathtools as mt
from copy import deepcopy

deg2rad = np.pi / 180.0
abcABC = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")

# reloading: https://towardsdatascience.com/introducing-reloading-never-re-run-your-python-code-again-to-print-more-details-374bee33473d

class MicroStatePrivate:

    debug = False
    tab = "\t\t"
    check = True
    thr = 0.1
    check_orth = True
    fmt = "%20.12e"
    # output files
    ofile = {"energy":"energy.txt",\
             "phases":"phases.txt",\
             "occupations":"occupations.txt",\
             "A-amplitudes":"A-amplitudes.txt",\
             "B-amplitudes":"B-amplitudes.txt",\
             "violin":"violin.csv"}
    
    smallest_float = np.nextafter(0,1)
    

class MicroState:

    #@reloading
    def add(self,*argv,**argc):
        try :
            print("Adding new attributes ...\n")
            # # empty object
            # temp = MicroState()
            # # give it the attributes
            # temp = merge_attributes(temp,self)
            # # read new attrbitues
            temp = deepcopy(self)
            MicroState.__init__(temp,*argv,**argc)
        except:
            raise ValueError("Error in __int__")
        
        print("Merging new attributes ...")
        temp = merge_attributes(self,temp)
        print("Merging completed :)\n")
        #return temp
        
    #@reloading
    def __init__(self,options=None,what=None,toread=None):

        print("Initializing object of type 'MicroState' ...")

        if options is None :
            options= {}

        if type(options) == dict :
            options = Dict2Obj(options)
        
        attribute_names  = [ "relaxed", "positions", "displacement", "velocities", "cells", "types" ]
        attribute_names += [ "eigvals", "dynmat", "eigvec", "modes", "ortho_modes", "masses" ]
        attribute_names += [ "Nmodes", "Nconf" ]
        attribute_names += [ "energy", "Aamplitudes", "Bamplitudes", "properties" ]

        for name in attribute_names:
            if not hasattr(self,name):
                setattr(self, name, None)

        if what is None :
            what = ""
        if toread is None :
            toread = list()

        if what == "vib":
            toread += [ "masses",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals"]            

        if what == "proj-on-vib-modes" :
            toread += [ "relaxed",\
                        "masses",\
                        "positions",\
                        "displacements",\
                        "velocities",\
                        #"modes",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals",\
                        #"dynmat",\
                        "properties"]
            
        if what == "plot-vib-modes-energy" :
            toread += [ "eigvals",\
                        "energy",\
                        "A-amplitudes",\
                        "properties"]
            
        if what == "generate-thermal-state":
            toread += [ "relaxed",\
                        "masses",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals",\
                        "atoms"]
            
        print("\nProperties to be read:")
        for k in toread:
            print("{:s}".format(MicroStatePrivate.tab),k)

        ###################
        #                 # 
        #  start reading  #
        #                 #
        ###################

        if "relaxed" in toread:
            ###
            # reading original position
            print("{:s}reading original/relaxed position from file '{:s}'".format(MicroStatePrivate.tab,options.relaxed))
            tmp = io.read(options.relaxed)
            atoms = tmp.get_chemical_symbols()
            relaxed = tmp.positions
            if relaxed.shape[1] != 3 :
                raise ValueError("the relaxed configurations do not have 3 components")
            self.Nmodes = relaxed.shape[0] * 3
            self.relaxed = relaxed.flatten()


        if "masses" in toread:
    
            if not hasattr(options, 'masses') or options.masses is None :
                if not os.path.isdir(options.modes):
                    raise ValueError("'--modes' should be a folder")            
                file = get_one_file_in_folder(folder=options.modes,ext=".masses")
                
            else :
                file = options.masses
            print("{:s}reading masses from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.masses = np.loadtxt(file)

            # no longer supported
            # if len(masses) == len(relaxed.positions) :
            #     # set masses
            #     M = np.zeros((3 * len(masses)), float)
            #     M[ 0 : 3 * len(masses) : 3] = masses
            #     M[ 1 : 3 * len(masses) : 3] = masses
            #     M[ 2 : 3 * len(masses) : 3] = masses
            #     masses = M

            # elif len(masses) != 3 * len(relaxed.positions):            
            #     raise ValueError("wrong number of nuclear masses")
                        
            # positions
            # relaxed = relaxed.positions
            # Nmodes = relaxed.shape[0] * 3

        if "positions" in toread:

            print("{:s}reading positions from file '{:s}'".format(MicroStatePrivate.tab,options.positions))
            positions = io.read(options.positions,index=":")
            tmp = positions[0]
            atoms = tmp.get_chemical_symbols()
            Nconf = len(positions) 

            for n in range(Nconf):
                positions[n] = positions[n].positions.flatten()

            if self.Nmodes is None :
                self.Nmodes = len(positions[0])
            for i in range(Nconf):
                if np.asarray( len(positions[i]) != self.Nmodes) :
                    raise ValueError("some configurations do not have the correct shape")

            self.positions = np.asarray(positions)
            self.Nconf = Nconf

        if "types" in toread:

            print("{:s}reading atomic types from file '{:s}'".format(MicroStatePrivate.tab,options.types))
            positions = io.read(options.positions,index=":")
            self.types = [ system.get_chemical_symbols() for system in positions ]


        if "cells" in toread :
            print("{:s}reading cells (for each configuration) from file '{:s}'".format(MicroStatePrivate.tab,options.cells))

            comments = read_comments_xyz(options.cells)
            cells = [ abcABC.search(comment) for comment in comments ]
            self.cell = np.zeros((len(cells),3,3))
            for n,cell in enumerate(cells):
                a, b, c = [float(x) for x in cell.group(1).split()[:3]]
                alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
                self.cell[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

            
        if "atoms" in toread:
            self.atoms = atoms


        if "displacements" in toread:

            if self.positions is None :
                raise ValueError("'positions' not defined")
            if self.relaxed is None:
                raise ValueError("'relaxed' not defined")            
            self.displacements = np.asarray(self.positions) - np.asarray(self.relaxed)


        if "velocities" in toread:

            if options.velocities is None:
                print("{:s}setting velocities to zero".format(MicroStatePrivate.tab))
                if self.positions is None :
                    raise ValueError("'positions' not defined")
                self.velocities = np.zeros(self.positions.shape)

            else :
                print("{:s}reading velocities from file '{:s}'".format(MicroStatePrivate.tab,options.velocities))
                velocities = io.read(options.velocities,index=":")
                Nvel = len(velocities)
                print("{:s}read {:d} velocities".format(MicroStatePrivate.tab,Nvel))
                if self.Nconf is not None :
                    if Nvel != self.Nconf :
                        raise ValueError("number of velocities and positions configuration are different")
                for n in range(Nvel):
                    velocities[n] = velocities[n].positions.flatten()
                self.velocities = np.asarray(velocities)


        if "ortho_modes" in toread:   

            if not os.path.isdir(options.modes):
                raise ValueError("'--modes' should be a folder")

            print("{:s}searching for '*.mode' file in folder '{:s}'".format(MicroStatePrivate.tab,options.modes))            
            file = get_one_file_in_folder(folder=options.modes,ext=".mode")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            modes = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = modes.shape[0]
            if modes.shape[0] != self.Nmodes or modes.shape[1] != self.Nmodes :
                raise ValueError("vibrational modes matrix with wrong size")
            if modes.shape[0] != modes.shape[1]:
                raise ValueError("vibrational modes matrix is not square")
            self.ortho_modes = modes


        if "eigvec" in toread:

            if not os.path.isdir(options.modes):
                raise ValueError("'--modes' should be a folder")
            
            file = get_one_file_in_folder(folder=options.modes,ext=".eigvec")
            print("{:s}reading eigenvectors from file '{:s}'".format(MicroStatePrivate.tab,file))
            eigvec = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = eigvec.shape[0]
            if eigvec.shape[0] != self.Nmodes or eigvec.shape[1] != self.Nmodes:
                raise ValueError("eigenvectors matrix with wrong size")
                
            # check that the eigenvectors are orthogonal (they could not be so)
            if MicroStatePrivate.check_orth :                
                print("{:s}checking that the eigenvectors are orthonormal, i.e. M @ M^t = Id".format(MicroStatePrivate.tab))
                res = np.linalg.norm(eigvec @ eigvec.T - np.eye(self.Nmodes))
                print("{:s} | M @ M^t - Id | = {:>20.12e}".format(MicroStatePrivate.tab,res))
                if res > MicroStatePrivate.thr :
                    raise ValueError("the eigenvectors are not orthonormal")            
            self.eigvec = eigvec


        if "modes" in toread:   
            if self.eigvec is None :
                raise ValueError("'eigvec' not defined")
            if self.masses is None:
                raise ValueError("'masses' not defined")   
            self.modes = MicroState.diag_matrix(self.masses,"-1/2") @ self.eigvec


        if "proj" in toread:   
            if self.eigvec is None :
                raise ValueError("'eigvec' not defined")
            if self.masses is None:
                raise ValueError("'masses' not defined")   
            self.proj = self.eigvec.T @ MicroState.diag_matrix(self.masses,"1/2")


        # I read the full hessian
        if "hess" in toread:
        
            if not os.path.isdir(options.modes):
                    raise ValueError("'--modes' should be a folder")   
                
            file = get_one_file_in_folder(folder=options.modes,ext="_full.hess")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            hess = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = hess.shape[0]
            if hess.shape[0] != self.Nmodes or hess.shape[1] != self.Nmodes:
                raise ValueError("hessian matrix with wrong size")           
            self.hess = hess


        # # pay attention: I never use it, so it has still to be debugged
        # if "full_hess" in toread:

        #     if not os.path.isdir(options.modes):
        #             raise ValueError("'--modes' should be a folder")   
            
        #     file = get_one_file_in_folder(folder=options.modes,ext="_full.hess")
        #     print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
        #     full_hess = np.loadtxt(file)
        #     if self.Nmodes is not None :
        #         if hess.shape[0] != self.Nmodes or hess.shape[1] != self.Nmodes:
        #             raise ValueError("full hessian matrix with wrong size")       
        #     else :
        #         self.Nmodes = len       
        #     self.full_hess = full_hess
            

        if "eigvals" in toread:

            if not os.path.isdir(options.modes):
                    raise ValueError("'--modes' should be a folder")   
            
            file = get_one_file_in_folder(folder=options.modes,ext=".eigval")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            eigvals = np.loadtxt(file)
            if self.Nmodes is not None :
                if len(eigvals) != self.Nmodes:
                    raise ValueError("eigenvalues array with wrong size")
            else :
                self.Nmodes = len(eigvals)
            self.eigvals = eigvals
            if np.any( self.eigvals < 0.0 ):
                print("{:s}!**Warning**: some eigenvalues are negative, setting them to (nearly) zero".format(MicroStatePrivate.tab))
                self.eigvals = np.asarray( [ MicroStatePrivate.smallest_float if i < 0.0 else i for i in self.eigvals ] )
            
        if "dynmat" in toread:

            if not os.path.isdir(options.modes):
                    raise ValueError("'--modes' should be a folder")   
            
            file = get_one_file_in_folder(folder=options.modes,ext=".dynmat")
            print("{:s}reading the dynamical matrix from file '{:s}'".format(MicroStatePrivate.tab,file))
            dynmat = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = dynmat.shape[0]
            if dynmat.shape[0] != self.Nmodes or dynmat.shape[1] != self.Nmodes:
                raise ValueError("dynamical matrix with wrong size")
            self.dynmat = dynmat
               
        # if MicroStatePrivate.check :
        #     print("\n{:s}Let's do a little test".format(MicroStatePrivate.tab))
        #     # mode      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".mode"))
        #     # dynmat    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".dynmat"))
        #     # full_hess = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext="_full.hess"))
        #     # eigvals    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigvals"))
        #     # eigvec    = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".eigvec"))
        #     # hess      = np.loadtxt(get_one_file_in_folder(folder=options.modes,ext=".hess"))
            
        #     if np.all(a is not None for a in [self.dynmat,self.eigvec,self.eigvals]):
        #         print("{:s}checking that D@V = E@V".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(self.dynmat @ self.eigvec - self.eigvals @ self.eigvec).sum())
        #         print("{:s} | D@V - E@V | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #     if np.all(a is not None for a in [self.modes,self.eigvals]):
        #         eigsys = np.linalg.eigh(self.modes)

        #         print("{:s}checking that eigvec(M) = M".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(eigsys[1] - self.modes).flatten().sum())
        #         print("{:s} | eigvec(H) - M | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #         print("{:s}checking that eigvals(H) = E".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square( np.sort(eigsys[0]) - np.sort(self.eigvals)).sum())
        #         print("{:s} | eigvec(H) - E | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #         print("{:s}checking that H@eigvec(H) = eigvals(H)@eigvec(H)".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(eigsys[0] - self.eigvals).sum())
        #         print("{:s} | eigvec(H) - E | = {:>20.12e}".format(MicroStatePrivate.tab,res))
            
        if "energy" in toread:
    
            file = output_file(options.output,MicroStatePrivate.ofile["energy"])
            print("{:s}reading energy from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.energy = np.loadtxt(file)


        if "A-amplitudes" in toread:

            file = output_file(options.output,MicroStatePrivate.ofile["A-amplitudes"])
            print("{:s}reading A-amplitudes from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.Aamplitudes = np.loadtxt(file)
            
            if self.energy is not None:

                if np.any(self.Aamplitudes.shape != self.energy.shape):
                    raise ValueError("energy and A-amplitudes matrix size do not match")
            
        # if "time" in toread:
        #     if options.property is None:
        #         raise ValueError("The file with the system (time-dependent) properties is not defined")
            
        #     t,u = getproperty(options.property,["time"])
        #     self.time  = t["time"]
        #     self.units = u["time"]

        #     if u["time"] not in ["a.u.","atomic_unit"]:
        #         print("{:s}'time' is not in 'atomic units' but in '{:s}'".format(MicroStatePrivate.tab,u["time"]))
        #         factor = unit_to_internal("time","femtosecond",1)
        #         print("{:s}converting 'time' to 'atomic units' by multiplication for {:>14.10e}".format(MicroStatePrivate.tab,factor))
        #         self.time *= factor
        #         self.units = "a.u."


        if "properties" in toread:
            if options.property is None:
                raise ValueError("The file with the system (time-dependent) properties is not defined")
            
            header = get_property_header(options.property,search=True)
            p,u = getproperty(options.property,header)
            self.header  = get_property_header(options.property,search=False)
            self.properties  = p
            self.units = u
            

        for name in attribute_names:
            if getattr(self, name) is None:
                delattr(self, name)

        print("\nInitialization completed :)") 
        pass

    @staticmethod
    def project_displacement(displ,proj):
        return proj @ displ
    
    @staticmethod
    def project_velocities(vel,proj,eigvals):
        # N = len(eigvals)
        # omega_inv = np.zeros((N,N))
        # np.fill_diagonal(omega_inv,1.0/np.sqrt(eigvals))
        # return np.nan_to_num(MicroState.diag_matrix(eigvals,"-1/2") @ proj @ vel,0.0)
        return MicroState.diag_matrix(eigvals,"-1/2") @ proj @ vel
    
    @staticmethod
    def potential_energy_per_mode(proj_displ,eigvals): #,hess=None,check=False):
        """return an array with the potential energy of each vibrational mode"""        
        return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ
    
    @staticmethod
    def kinetic_energy_per_mode(proj_vel,eigvals): #,check=False):
        """return an array with the kinetic energy of each vibrational mode"""        
        return 0.5 * ( np.square(proj_vel).T * eigvals ).T #, 0.5 * ( proj_vel * eigvals ) * identity @ ( eigvals * proj_vel )

    @staticmethod
    def diag_matrix(M,exp):
        out = np.eye(len(M))        
        if exp == "-1":
            np.fill_diagonal(out,1.0/M)
        elif exp == "1/2":
            np.fill_diagonal(out,np.sqrt(M))
        elif exp == "-1/2":
            np.fill_diagonal(out,1.0/np.sqrt(M))
        else :
            raise ValueError("'exp' value not allowed")
        return out       

    @staticmethod
    def A2B(A,N,M,E):
        """
        purpose:
            convert the A-amplitude [length x mass^{-1/2}] into B-amplitudes [length]

        input :
            A : A-amplitudes
            N : normal modes (normalized)
            M : masses
            E : eigevectors (of the dynamical matrix)

        output:
            B : B-amplitudes
        """
        
        # print("A shape : ",A.shape)
        # print("N shape : ",N.shape)
        # print("M shape : ",M.shape)
        # print("E shape : ",E.shape)

        B = np.diag( np.linalg.inv(N) @ MicroState.diag_matrix(M,"-1/2") @ E ) * A
        # print("B shape : ",B.shape)
        return B

    def project_on_cartesian_coordinates(self,Aamp=None,phases=None,inplace=True):
        
        if Aamp is None :
            Aamp = self.Aamplitudes
        if phases is None :
            phases = self.phases

        if len(Aamp.shape) == 1 :
            Aamp = Aamp.reshape(1,-1)
        if len(phases.shape) == 1 :
            phases = phases.reshape(1,-1)

        if "time" in self.properties and self.properties is not None:
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(Aamp))
            
        phi = np.outer(np.sqrt( self.eigvals) , time).T
        c = + Aamp * np.cos( phi + phases )
        s = - Aamp * np.sin( phi + phases)
        
        #np.linalg.inv(self.proj)
        invproj = MicroState.diag_matrix(self.masses,"-1/2") @ self.eigvec 

        deltaR = ( invproj @ c.T ).T
        v = ( invproj @ MicroState.diag_matrix(self.eigvals,"1/2") @ s.T ).T

        N = len(deltaR)
        positions = np.full((N,len(self.relaxed)),np.nan)
        for i in range(N):
            positions[i] = deltaR[i] + self.relaxed

        if inplace :
            self.displacements = deltaR
            self.velocities = v
            self.positions = positions

        return { "displacements":deltaR,\
                 "velocities":v,\
                 "positions":positions }

    # @reloading
    def project_on_vibrational_modes(self,deltaR=None,v=None,inplace=True):

        if deltaR is None :
            deltaR = self.displacements
        if v is None :
            v = self.velocities

        if len(deltaR.shape) == 1 :
            deltaR = deltaR.reshape(1,-1)
        if len(v.shape) == 1 :
            v = v.reshape(1,-1)
       
        # arrays = [  self.displacements,\
        #             self.velocities,\
        #             #self.modes, \
        #             #self.hess, \
        #             self.eigvals, \
        #             #self.Nmodes, \
        #             #self.dynmat, \
        #             #self.eigvec, \
        #             #self.Nconf,\
        #             #self.masses,\
        #             self.ortho_modes,\
        #             self.proj,\
        #             self.time ]
        
        # if np.any( arrays is None ) :
        #     raise ValueError("'compute': some arrays are missing")

        # c = ( self.proj @ deltaR.T )
        # s = ( MicroState.diag_matrix(self.eigvals,"-1/2") @ self.proj @ v.T )
        # A = np.sqrt(np.square(c) + np.square(s))
        
        proj_displ = MicroState.project_displacement(deltaR.T,self.proj).T
        proj_vel   = MicroState.project_velocities  (v.T,   self.proj, self.eigvals).T
        A2 = ( np.square(proj_displ) + np.square(proj_vel) )
        energy = ( self.eigvals * A2 / 2.0 ) # w^2 A^2 / 2
        energy [ energy == np.inf ] = np.nan
        normalized_energy = ( ( self.Nmodes - 3 ) * energy.T / energy.sum(axis=1).T ).T
        Aamplitudes = np.sqrt(A2)

        # print(norm(proj_displ-c))
        # print(norm(proj_vel-s))
        
        # Vs = MicroState.potential_energy_per_mode(proj_displ,self.eigvals)
        # Ks = MicroState.kinetic_energy_per_mode  (proj_vel,  self.eigvals)
        # Es = Vs + Ks        
        # print(norm(energy-Es.T))

        # self.energy = self.occupations = self.phases = self.Aamplitudes = self.Bamplitudes = None 
    
        # energy = Es.T
        occupations = energy / np.sqrt( self.eigvals) # - 0.5 # hbar = 1 in a.u.
        # A  = np.sqrt( 2 * Es.T / self.eigvals  )
        # print(norm(A-Aamplitudes))

        Bamplitudes = MicroState.A2B(A=Aamplitudes,\
                                    N=self.ortho_modes,\
                                    M=self.masses,\
                                    E=self.eigvec)
        
        if "time" in self.properties and self.properties is not None:
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(Bamplitudes))
        phases = np.arctan2(-proj_vel,proj_displ) - np.outer(np.sqrt( self.eigvals) , time).T

        out = {"energy": energy,\
               "norm-energy": normalized_energy,\
               "occupations": occupations,\
               "phases": phases,\
               "A-amplitudes": Aamplitudes,\
               "B-amplitudes": Bamplitudes}
        
        if inplace:
            self.energy = energy
            self.occupations = occupations
            self.phases = phases
            self.Aamplitudes = Aamplitudes
            self.Bamplitudes = Bamplitudes
            self.normalized_energy = normalized_energy

        if MicroStatePrivate.debug : test = self.project_on_cartesian_coordinates(inplace=False)
        if MicroStatePrivate.debug : print(norm(test["positions"] - self.positions))
        if MicroStatePrivate.debug : print(norm(test["velocities"] - self.velocities))
        if MicroStatePrivate.debug : print(norm(test["displacements"] - self.displacements))

        return out
    
    def save2xyz(self,what,file=None,name=None,folder=None,atoms=None):

        if file is None:
            file = output_file(folder,MicroStatePrivate.ofile[name])
        if atoms is None :
            atoms = self.atoms

        if len(what.shape) == 1 : # just one configuration, NOT correctly formatted

            what = what.reshape((-1,3))
            return self.save2xyz(what,file,name,folder,atoms)
        
        elif len(what.shape) == 2 : 

            if what.shape[1] != 3 : # many configurations
                what = what.reshape((len(what),-1,3))
                return self.save2xyz(what,file,name,folder,atoms)
            
            else : # just one configurations, correctly formatted
                return self.save2xyz(np.asarray([what]),file,name,folder,atoms)

        elif len(what.shape) == 3 :

            Na = what.shape[1]
            if what.shape[2] != 3 :
                raise ValueError("wrong shape")
            
            with open(file,"w") as f :
                
                for i in range(what.shape[0]):
                    pos = what[i,:,:]
                    f.write(str(Na)+"\n")
                    f.write("# configuration {:d}\n".format(i))
                    for ii in range(Na):
                        f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))
            return


    @staticmethod
    def save2txt(what,file=None,name=None,folder=None):
        if file is None:
            file = output_file(folder,MicroStatePrivate.ofile[name])
        print("{:s}saving {:s} to file '{:s}'".format(MicroStatePrivate.tab,name,file))
        np.savetxt(file,what, fmt=MicroStatePrivate.fmt)
        pass

    def savefiles(self,folder,what):

        if what == "proj-on-vib-modes":
            MicroState.save2txt(what=self.energy,name="energy",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["energy"])
            # print("{:s}saving energy to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.energy.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.occupations,name="occupations",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["occupations"])
            # print("{:s}saving occupations to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.occupations.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.phases,name="phases",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["phases"])
            # print("{:s}saving phases to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.phases.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.Aamplitudes,name="A-amplitudes",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["Aamp"])
            # print("{:s}saving A-amplitudes to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.Aamplitudes.T,fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.Bamplitudes,name="B-amplitudes",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["Bamp"])
            # print("{:s}saving B-amplitudes to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.Bamplitudes.T,fmt=MicroStatePrivate.fmt)

        pass

    def plot(self,options):

        if "time" in self.properties and self.properties is not None:
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(self.Aamplitudes))

        if options.t_min > 0 :            
            print("\tSkipping the {:d} {:s}".format(options.t_min,self.units))
            i = np.where( self.time >= options.t_min )[0][0]
            print("\tthen skipping the first {:d} MD steps".format(i))
            self.Aamplitudes = self.Aamplitudes[i:,:]
            self.energy = self.energy[i:,:] 
            time   = time[i:]

        Ndof = self.Aamplitudes.shape[1]
        normalization = self.energy.sum(axis=1) / ( Ndof - 3 )

        normalized_occupations = np.zeros(self.Aamplitudes.shape)
        for i in range(Ndof):
            normalized_occupations[:,i] = np.square(self.Aamplitudes[:,i])  * self.eigvals[i] / ( 2*normalization[i] )

        fig, ax = plt.subplots(figsize=(10,6))

        factor = unit_to_user("time","picosecond",1)
        time = time*factor
        ax.plot(time,normalized_occupations)

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("time (ps)")
        ax.set_xlim(min(time),max(time))
        ylim = ax.get_ylim()
        #ax.set_ylim(0,ylim[1])
        ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        plt.savefig(options.plot)

        ###
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        mean = np.mean(normalized_occupations,axis=0)
        std = np.std(normalized_occupations,axis=0)
        if len(mean) != Ndof or len(std) != Ndof:
            raise ValueError("wrong array size for barplot")

        fig, ax = plt.subplots(figsize=(10,6))
        w = np.sqrt(self.eigvals) * unit_to_user("frequency","thz",1)
        # ax.scatter(x=w,y=mean,color="navy")
        ax.errorbar(x=w,y=mean,yerr=std,color="red",ecolor="navy",fmt="o")

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("$\\omega$ (THz)")
        
        #ax.set_xlim(min(self.time),max(self.time))
        xlim = ax.get_xlim()
        ax.hlines(1.0,xlim[0],xlim[1],linestyle="dashed",color="black",alpha=0.5)
        ax.set_xlim(*xlim)
        # ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        tmp = os.path.splitext(options.plot)
        file = "{:s}.{:s}{:s}".format(tmp[0],"mean-std",tmp[1])
        # plt.show()
        plt.savefig(file)

        df = pd.DataFrame()
        df["w [THz]"] = w
        df["mean"] = mean
        df["std"] = std
        file = file = output_file(options.output,MicroStatePrivate.ofile["violin"])
        df.to_csv(file,index=False,float_format="%22.12f")

        pass

    def get_position_template(self,flattened=True):
        if flattened:
            return np.full(self.relaxed.shape,np.nan)
        else :
            return self.get_position_template(flattened=False).reshape((-1,3))
        
    def get_position_template_shape(self,flattened=True):
        if flattened:
            return list(self.relaxed.shape)
        else :
            return list(len(self.relaxed)/3,3)
    
    def generate_thermal_state(self,T,randomfile,N=1,save=True,read=True,unit="kelvin"):

        if unit == "kelvin":
            factor = unit_to_internal("temperature","kelvin",1)
            print("{:s}converting temperature from 'kelvin' to 'atomic units' by multiplication for {:>14.10e}".format(MicroStatePrivate.tab,factor))
            T *= factor

        if N <= 0 :
            raise ValueError("'N' has to be greater than 0")
        
        # set the random number generator state
        if read :
            if not randomfile.endswith(".pickle") :
                randomfile += ".pickle"
        
            if os.path.exists(randomfile):
                try :
                    with open(randomfile, 'rb') as f:
                        state = pickle.load(f)
                    rand.set_state(state)
                except:
                    print("{:s}file '{:s}' supposed to contain the (pseudo-)random number generator state is empty".format(MicroStatePrivate.tab,randomfile))

        # call generate_thermal_state N times
        if N > 1 :
            tmp = self.get_position_template_shape()
            r = np.full(tmp*N,np.nan,dtype=object)
            v = r.copy()

            if hasattr(T,"__len__"):
                if len(T) != N :
                    raise ValueError("'N' and 'T' must have the same length")
            else :
                T = np.full(N,T)

            for i in range(N):
                r[i],v[i] = self.generate_thermal_state(T=T[i],randomfile=None,N=1,save=False,read=False,unit="a.u.")

        else:
            # r = self.get_position_template()
            # v = self.get_position_template()

            # generate random phases
            phases = rand.rand(len(self.relaxed))*2*np.pi

            # generate A-amplitudues according to the Equipartition Theorem
            Amplitudes = np.full(self.Nmodes,0.0) # Nmodes = number degrees of freedom
            #total_energy = self.Nmodes * T # since k_B = 1

            # T = E_tot / N = A^2 * w^2 / 2
            # sqrt( 2 T / w^2 ) = A 
            Amplitudes[3:] = np.sqrt( 2 * T / self.eigvals[3:] )

            out = self.project_on_cartesian_coordinates(Aamp=Amplitudes,\
                                                        phases=phases,\
                                                        inplace=False)
            
            r = out["positions"]
            v = out["velocities"]

            # test = self.project_on_vibrational_modes(deltaR=out["displacements"],v=v,inplace=False)
                

        if save:
            with open(randomfile, 'wb') as f:
                print("{:s}Saving the (pseudo-)random number generator state to file '{:s}'".format(MicroStatePrivate.tab,randomfile))
                s = rand.get_state()
                pickle.dump(s, f)

        return r,v
    
    def get_properties_as_dataframe(self):

        import pandas as pd
        p = pd.DataFrame(data=self.properties,columns=self.header)
        return p
    
    def show(self):
        '''show the attributes of the class'''
        print("Attributes of the object:")
        attribs = get_attributes(self)
        for a in attribs:
            print("{:s}".format(MicroStatePrivate.tab),a)

    def show_properties(self):
        '''show the properties of the class'''
        print("Properties of the object:")
        keys = list(self.properties.keys())
        size = [None]*len(keys)
        for n,k in enumerate(keys):
            tmp = list(self.properties[k].shape[1:])
            if len(tmp) == 0 :
                size[n] = 1
            elif len(tmp) == 1:
                size[n] = tmp[0]
            else :
                size[n] = tmp
        df = pd.DataFrame(columns=["name","unit","shape"])
        df["name"] = keys
        df["unit"] = [ self.units[k] for k in keys ]
        df["shape"] = size
        return df

    def convert_property(self,what,unit,family,inplace=True):
        # family = get_family(name)
        factor = convert(1,family,_from=self.units[what],_to=unit)
        if inplace :
            self.properties[what] = self.properties[what] * factor
            self.units[what] = unit
            return self.properties[what]
        else :
            return self.properties[what] * factor

    # @reloading
    def vibrational_analysis_summary(self):
        """ summary of the vibrational analysis"""
        print("Summary of the vibrational analysist:")
        cols = [ "eigvals [a.u.]" , "w [a.u.]", "w [THz]", "w [cm^-1]", "T [a.u.]", "T [ps]","E [a.u.]", "n [a.u.]"]
        df = pd.DataFrame(columns=cols)
        eigvals = self.eigvals.copy()
        eigvals [ eigvals == MicroStatePrivate.smallest_float ] = np.nan
        df["eigvals [a.u.]"] = eigvals
        df["w [a.u.]"]  = [ np.sqrt(i) if i > 0. else None for i in eigvals ]
        df["w [THz]"]   = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="thz")
        df["w [cm^-1]"] = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="inversecm")
        df["T [a.u.]"]  = 2*np.pi / df["w [a.u.]"]
        df["T [ps]"]    = convert(df["T [a.u.]"],"time",_from="atomic_unit",_to="picosecond")
        df["E [a.u.]"]  = self.energy.mean(axis=0)
        df["n [a.u.]"]  = self.occupations.mean(axis=0)
        return df

    # @reloading
    def to_ase(self,inplace=False,recompute=False):

        out = None
        if recompute or not hasattr(self,"ase"):
            out = [None]*self.Nconf
            N = np.arange(len(out))
            for n,t,p,c in zip(N,self.types,self.positions,self.cell):
                out[n] = Atoms(symbols=t, positions=p.reshape(-1,3), cell=c.T, pbc=True)

        if inplace and out is not None:
            self.ase = out
        elif hasattr(self,"ase"):
            out = self.ase

        return out

    # @reloading
    @staticmethod
    def save(obj,file):
        print("Saving object to file '{:s}'".format(file))
        with open(file, 'wb') as f:
            pickle.dump(obj,f)
        pass

    # @reloading
    @staticmethod
    def load(file):
        print("Loading object from file '{:s}'".format(file))
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj
