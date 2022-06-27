#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This program takes 'simulation.pos_*.xyz' files, and calculates
    g(z) distribution, where z is the distance along z between a surface and a molecule.
    The surface and molecule(s) are defined ad hoc,
    please see the code and adjust, it is necessary.

    ATTENTION: handling of units is not ideal:
    no conversions, it just takes numbers from xyz files and processes them.
    It says angstroms just because I always had angstroms.

    v2.0 can use several directories at once
    v2.1 uses Kahan summation for averaging, I hope it will reduce the error.
         https://stackoverflow.com/questions/42816678/fsum-for-numpy-arrays-stable-summation
    v2.2 adds accounting for the number of beads and for the number of independent frames.
"""

import sys, os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from ipi.utils.io import read_file_raw

nbins = 650
zmin = 0.
zmax = 6.5

arginfo = ("Possibe arguments:\n"
          "\t-species <species> - atomic species to calculate g(z)\n"
          "\teither '-d' or '-here':\n"
          "\t\t-d <dir1> <Nskip1> <dir2> <Nskip2> ... <dirN> <NskipN> - directories to take trajectories from,\n"
          "\t\t\tand numbers of step to skip in the beginning"
          "\t\t-here to use current directory\n"
          "\t\tNote that dir1 will be used for naming output files.\n"
          "optionally:\n"
          "\t-nbins <nbins> - number of bins for histogram [%i]\n"
          "\t-zmin <zmin> - minimal z for the histogram [%g]\n"
          "\t-zmax <zmax> - maximal z for the histogram [%g angstrom]\n" % (nbins, zmin, zmax))

def kahan_sum(a, axis=0):
    """ Kahan summation of the numpy array along an axis.
        https://stackoverflow.com/questions/42816678/fsum-for-numpy-arrays-stable-summation
        I try to reduce the summation error in averaging.
    """
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # https://stackoverflow.com/a/42817610/353337
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()
    return s


def parse_frame(rr, histo, rawdata, bottom_hydrogens, species, framecounter):
    """ Parses single frame and adds found atoms to the histogram provided as an argument.
        arguments:
            rr      - frame read by read_file_raw()
            histo   - histogram to add points to
            species - which species to search
    """
    surface = []
    hydrogens = []
    bottom_h_tmp = []       # To have the 1st peak exactly, without overlap with the second one.
    rawdataline = []    # list of all h values within one frame
    for i in range(rr['natoms']):
        if rr['names'][i] == 'Rh':
           surface.append(rr['data'][3*i + 2])
        elif rr['names'][i] == species:
            hydrogens.append(rr['data'][3*i + 2])
        # very ad-hoc solution , which relies on the fact
        # that atoms are ordered in the same way everywhere:
        if i in [12, 18, 20, 24, 29, 32]:
            bottom_h_tmp.append(rr['data'][3*i + 2])

    if len(surface) == 0:
        raise RuntimeError("Rhodium not found")

    if len(surface) % 4 != 0:
        raise RuntimeError("Does the system have not 4N surface layers?")

    surface = sorted(surface, key=float)
    topsurface = np.asarray(surface[len(surface)*3//4 :], dtype=np.float64)
    z0 = np.mean(topsurface)

    for h in hydrogens:
        d = h-z0
        if d > zmax:
            raise RuntimeError("Some atoms are above zmax (distance to surface %f Angstrom found),\nconsider readjusting histogram settings." % d)
        elif d < 0:
            raise RuntimeError("Some atoms are below the surface.")

        rawdata.append([framecounter, d])
        histo[int(np.floor(d / binsize))] += 1

    for h in bottom_h_tmp:
        d = h-z0
        bottom_hydrogens.append(d)

    return histo, rawdata, bottom_hydrogens


#============================ Algorithm starts here ======================================
rootdir = os.getcwd()
dirlist = []
nskiplist = []      # A list to store nskip for each directory
#interval   = 60     # Read one step each 30 femtoseconds (60 steps)
corrtime = 600      # Correlation time in steps. I take w.f. correlation time, which is ~300 fs.
zerostep   = 0
zerocell   = []     # cell in the first read frame
zeronatoms = 0      # number of atoms in the first read frame
step       = -1

nskip = None
species = None

if len(sys.argv) == 1:
    print("\'g_z_from_xyz.py\' takes 'simulation.pos_*.xyz' files, and calculates\n"
          "g(z) distribution (z is z-coordinate).\n"
          "v2.0 can parse multiple directories.\n"
          "v2.1 uses Kahan summation for averaging, I hope it will reduce the error.\n"
          "v2.2 adds accounting for the number of beads and for the number of independent frames.")
    print(arginfo)
    exit(-1)
else:
    for i, arg in enumerate(sys.argv):
        if arg == '-species':
            species = sys.argv[i+1]
        if arg == '-d':
            for j in range(i+1, len(sys.argv), 2):
                if sys.argv[j][0] == '-':
                    break
                else:
                    dirlist.append(sys.argv[j])
                    if sys.argv[j+1][0] == '-':
                        print ("ERROR: Each <dir> entry should have its own nskip.")
                        print(arginfo)
                        exit(-1)
                    else:
                        nskiplist.append(int(sys.argv[j+1]))
        if arg == '-here':
            dirlist.append(rootdir)
            nskiplist.append(int(sys.argv[i+1]))
        if arg == '-nbins':
            nbins = int(sys.argv[i+1])
        if arg == '-zmin':
            zmin = float(sys.argv[i+1])
        if arg == '-zmax':
            zmax = float(sys.argv[i+1])
if ('-d' in sys.argv) and ('-here' in sys.argv):
    print("ERROR:\t\t-d and -here keys are incompatible!")
    exit(-1)

if (nskiplist == []) or (species == None) or (dirlist == []):
    print("ERROR: incorrect arguments.")
    print(arginfo)
    exit(-1)
if len(dirlist) != len(nskiplist):
    print("ERROR: incorrect arguments. Each dir should have nskip.")
    print(arginfo)
    exit(-1)


histo = np.zeros(nbins, dtype=float)
rawdata = []            # list of all h-z0 values
bottom_hydrogens = []   # list of all bottom hydrogens' coordinates to have the 1st peak exactly, without overlap with the second one
assert zmax > zmin
binsize = (zmax - zmin) / nbins
framecounter = 0    # Total number of frames
n_indep = 0         # Number of independent frames

for d, nskip in zip(dirlist, nskiplist):
    if d[0] == '/':
        dd = d
        os.chdir(dd)
    else:
        dd = os.path.join(rootdir, d)
        os.chdir(dd)
    # Parse one directory:
    nbeads=0            # Number of beads in current dir
    nframes_local=0     # Number of frames in current dir
    for file in sorted(os.listdir('.')):
        if fnmatch.fnmatch(file, 'simulation.pos_*.xyz'):
            print("Parsing %s/%s ..." % (dd, file))
            fdopen = open(file, 'r')

            # skip non-thermalized beginning:
            for i in range(nskip):
                read_file_raw('xyz', fdopen)

            # read the first step:
            rr = read_file_raw('xyz', fdopen)
            metainfo = rr['comment'].split()
            for i, word in enumerate(metainfo):
                if word == 'Step:':
                    zerostep = int(metainfo[i+1])
                    step = zerostep
                elif word == 'Bead:':
                    beadindex = int(metainfo[i+1])
            zeronatoms = rr['natoms']

            nbeads +=1
            histo, rawdata, bottom_hydrogens = parse_frame(rr, histo, rawdata, bottom_hydrogens, species, framecounter)
            nframes_local +=1

            while True:
                try:
                    # read the next step:
                    rr = read_file_raw('xyz', fdopen)
                    metainfo = rr['comment'].split()
                    for i, word in enumerate(metainfo):
                        if word == 'Step:':
                            step = int(metainfo[i+1])
                        elif word == 'Bead:':
                            if beadindex != int(metainfo[i+1]):
                                raise RuntimeError("The bead index within one file is not constant.")
                        if rr['natoms'] != zeronatoms:
                            raise RuntimeError("The number of atoms within one file is not constant.")
                    histo, rawdata, bottom_hydrogens = parse_frame(rr, histo, rawdata, bottom_hydrogens, species, framecounter)
                except EOFError:
                    break
                framecounter +=1
                nframes_local +=1
    # Finished parsing one directory
    if nbeads not in [1,6,12,16,24]:  # I never did different amount of beads so far.
        raise RuntimeError("Weird number of beads (%i) found in dir %s." % (nbeads, dd))
    if nframes_local % nbeads != 0:
        raise RuntimeError("nframes_local % nbeads != 0.\nBroken pos_*.xyz files?")
    # I add 1 because trajectories are independent
    # even if their length is less than 1 corrtime per traj.
    n_indep += 1 + nframes_local // nbeads // corrtime
    os.chdir(rootdir)

histo = histo / np.sum(histo)
x_axis = np.zeros(nbins, dtype=float)
for i in range(nbins):
    x_axis[i] = i*binsize

np.savetxt("g_z_%s_%s.dat" % (species, os.path.split(dirlist[0])[-1]),
           np.c_[x_axis, histo],
           fmt='%.4f',
           delimiter='    ',
           header="delta_Z_%s-Rh(111)_[A]    g(z)\n"
                  "Data gathered from the following directories: %s\n"
                  "Total number of frames:  %i"
           % (species,
              " ".join(line.strip() for line in sys.argv),
              framecounter
             )
          )

# Here you can output all distances from frames:
#rawdata = np.asarray(rawdata)
#np.savetxt('g_z_%s_%s.rawdata.dat'  % (species, os.path.split(dirlist[0])[-1]),
#           rawdata,
#           fmt='%i    %.5f',
#           header='# data gathered from the following directories: %s # Total number of frames:  %i'
#           % (", ".join(line.strip() for line in dirlist), framecounter))

# Here you can output all coordinates of the 6 bottom hydrogens:
bottom_hydrogens = np.asarray(bottom_hydrogens, dtype=np.float64)
mean_bot_h = kahan_sum(bottom_hydrogens) / len(bottom_hydrogens)

meansquare = kahan_sum(bottom_hydrogens **2) / len(bottom_hydrogens)
#bottom_stdev = np.std(bottom_hydrogens)
bottom_stdev = np.sqrt(meansquare - mean_bot_h**2)
bottom_error = bottom_stdev / np.sqrt(n_indep)

np.savetxt("g_z_%s_%s.bottom_hydrogens.dat"  % (species, os.path.split(dirlist[0])[-1]),
           bottom_hydrogens,
#           fmt='    %.5f    %.5f    %.5f',
           header="Z coordinates of 6 bottom hydrogens. ATTENTION: THIS PART RELIES ON THE ATOM ORDERING IN .XYZ FILES!\n"
                  "Data gathered by the following command: %s\n"
                  "Total number of frames:  %i\n"
                  "Mean z value is %f sigma is %f.\n"
                  "Assuming %i steps correlation time, error is %f. N_indep is %i"
                  % (" ".join(line.strip() for line in sys.argv),
                     framecounter,
                     mean_bot_h,
                     bottom_stdev,
                     corrtime,
                     bottom_error,
                     n_indep
                    )
          )
bottom_histo = np.histogram(bottom_hydrogens, bins=200, range=(1.4, 3.4), density=True)
#print(bottom_histo)

# ====================== Finding Full-width-half-maximum crossings with the line ======================
# ============= https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python ============
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]
# ======================================================================================================

hmx = half_max_x(bottom_histo[1][:-1], bottom_histo[0])
FWHM = hmx[1] - hmx[0]      # Full width at half maximum
FWHM_middle = (hmx[1] + hmx[0]) /2

# Output of a histogram:
np.savetxt("g_z_%s_%s.bottom_hydrogens.histo.dat"  % (species, os.path.split(dirlist[0])[-1]),
           np.column_stack((bottom_histo[1][:-1], bottom_histo[0])),
           header="Histogram of Z coordinates of 6 bottom hydrogens. ATTENTION: THIS PART RELIES ON THE ATOM ORDERING IN .XYZ FILES!\n"
                  "Data gathered by the following command: %s\n"
                  "Total number of frames:  %i\n"
                  "FWHM is %f, FWHM middle at %f A,\n"
                  "Mean value is %.3f, sigma is %.3f A.\n"
                  "Assuming %i steps correlation time, error is %f. N_indep is %i"
                  % (" ".join(line.strip() for line in sys.argv),
                     framecounter,
                     FWHM,
                     FWHM_middle,
                     mean_bot_h,
                     bottom_stdev,
                     corrtime,
                     bottom_error,
                     n_indep
                    )
          )

# Plot histogram:
#plt.xlabel('distance to surface [Angstrom]')
#plt.grid()
#plt.xlim((1., 6.))
#plt.plot(x_axis, histo)
#plt.show()
