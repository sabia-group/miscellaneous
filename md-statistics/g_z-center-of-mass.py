#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This program takes 'simulation.pos_*.xyz' files, and calculates
    g(z) distribution (z is z-coordinate) for the centers of mass,
    assuming that there are two molecules in appropriate format:
        6 carbons mol1
        6 carbons mol2
        12 H/D    mol1
        12 H/D    mol2
    It can use several directories at once.
"""

import sys, os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from ipi.utils.io import read_file_raw

nbins = 300
zmin = 2.0
zmax = 5.0

arginfo = (
    "Possibe arguments:\n"
    "\teither '-d' or '-here':\n"
    "\t\t-d <dir1> <Nskip1> <dir2> <Nskip2> ... <dirN> <NskipN> - directories to take trajectories from,\n"
    "\t\t\tand numbers of step to skip in the beginning"
    "\t\t-here <Nskip> to use current directory\n"
    "\t\tNote that dir1 will be used for naming output files (don't use trailing slash).\n"
    "optionally:\n"
    "\t-nbins <nbins> - number of bins for histogram [%i]\n"
    "\t-zmin <zmin> - minimal z for the histogram [%g]\n"
    "\t-zmax <zmax> - maximal z for the histogram [%g angstrom]\n" % (nbins, zmin, zmax)
)


def parse_frame(rr, histo, rawdata, framecounter):
    """Parses single frame and adds found atoms to the histogram provided as an argument.
    arguments:
        rr      - frame read by read_file_raw()
        histo   - histogram to add points to
    """
    surface = []
    mol1 = []
    mol2 = []
    mol1_masses = []
    mol2_masses = []

    for i in range(rr["natoms"]):
        if i < 6 and (rr["names"][i] == "C"):
            mol1.append(rr["data"][3 * i + 2])
            mol1_masses.append(12)
        elif 6 <= i < 12 and (rr["names"][i] == "C"):
            mol2.append(rr["data"][3 * i + 2])
            mol2_masses.append(12)
        elif 12 <= i < 24 and (rr["names"][i] in "H,D"):
            mol1.append(rr["data"][3 * i + 2])
            mol1_masses.append(1)
        elif 24 <= i < 36 and (rr["names"][i] in "H,D"):
            mol2.append(rr["data"][3 * i + 2])
            mol2_masses.append(1)
        elif rr["names"][i] == "Rh":
            surface.append(rr["data"][3 * i + 2])

    if len(surface) == 0:
        raise RuntimeError("Rhodium not found")

    if len(surface) % 4 != 0:
        raise RuntimeError("Does the system have not 4N surface layers?")

    if len(mol1) != 18 or len(mol2) != 18:
        raise RuntimeError("At least one molecule has not 18 atoms.")

    surface = sorted(surface, key=float)
    topsurface = np.asarray(surface[len(surface) * 3 // 4:], dtype=np.float64)
    z0 = np.mean(topsurface)

    # mol1
    com = 0  # Z of the center of mass
    for h, m in zip(mol1, mol1_masses):
        d = h - z0
        if d < 0:
            raise RuntimeError("Some atoms are below the surface.")
        com += h * m
    com = com / sum(mol1_masses)
    d = com - z0
    if d > zmax:
        raise RuntimeError(
            "Center of mass is above zmax (distance to surface %f Angstrom found),\nconsider readjusting histogram settings."
            % d
        )
    elif d < zmin:
        raise RuntimeError(
            "Center of mass is below zmin (distance to surface %f Angstrom found),\nconsider readjusting histogram settings."
            % d
        )
    rawdata.append([framecounter, d])
    histo[int(np.floor(d / binsize) - zmin / binsize)] += 1

    # mol2
    com = 0  # Z of the center of mass
    for h, m in zip(mol2, mol2_masses):
        d = h - z0
        if d < 0:
            raise RuntimeError("Some atoms are below the surface.")
        com += h * m
    com = com / sum(mol2_masses)
    d = com - z0
    if d > zmax:
        raise RuntimeError(
            "Center of mass is above zmax (distance to surface %f Angstrom found),\nconsider readjusting histogram settings."
            % d
        )
    elif d < zmin:
        raise RuntimeError(
            "Center of mass is below zmin (distance to surface %f Angstrom found),\nconsider readjusting histogram settings."
            % d
        )
    rawdata.append([framecounter, d])
    histo[int(np.floor(d / binsize) - zmin / binsize)] += 1

    return histo, rawdata


# ====================== Finding Full-width-half-maximum crossings with the line ======================
# ============= https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python ============
def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y):
    """Finds positions of half-maximum values, assuming a single peak
    """
    half = max(y) / 2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = signs[0:-2] != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    return [
        lin_interp(x, y, zero_crossings_i[0], half),
        lin_interp(x, y, zero_crossings_i[1], half),
    ]


# ============================ Algorithm starts here ======================================
rootdir = os.getcwd()
dirlist = []
nskiplist = []  # A list to store nskip for each directory
# interval   = 60     # Read one step each 30 femtoseconds (60 steps)
corrtime = (
    600  # Correlation time in steps. I take w.f. correlation time, which is ~300 fs.
)
zerostep = 0
zerocell = []  # cell in the first read frame
zeronatoms = 0  # number of atoms in the first read frame
step = -1


if len(sys.argv) == 1:
    print(
        "'g_z_center-of-mass.py' takes 'simulation.pos_*.xyz' files, and calculates\n"
        "g(z) distribution (z is z-coordinate) for the centers of mass,\n"
        "assuming that there are two molecules in appropriate format:\n"
        "\t6 carbons mol1\n"
        "\t6 carbons mol2\n"
        "\t12 H/D    mol1\n"
        "\t12 H/D    mol2\n"
        "It can use several directories at once."
    )
    print(arginfo)
    exit(-1)
else:
    for i, arg in enumerate(sys.argv):
        if arg == "-d":
            for j in range(i + 1, len(sys.argv), 2):
                if sys.argv[j][0] == "-":
                    break
                else:
                    dirlist.append(sys.argv[j])
                    if sys.argv[j + 1][0] == "-":
                        print("ERROR: Each <dir> entry should have its own nskip.")
                        print(arginfo)
                        exit(-1)
                    else:
                        nskiplist.append(int(sys.argv[j + 1]))
        if arg == "-here":
            dirlist.append(rootdir)
            nskiplist.append(int(sys.argv[i + 1]))
        if arg == "-nbins":
            nbins = int(sys.argv[i + 1])
        if arg == "-zmin":
            zmin = float(sys.argv[i + 1])
        if arg == "-zmax":
            zmax = float(sys.argv[i + 1])
if ("-d" in sys.argv) and ("-here" in sys.argv):
    print("ERROR:\t\t-d and -here keys are incompatible!")
    exit(-1)

if (nskiplist == []) or (dirlist == []):
    print("ERROR: incorrect arguments.")
    print(arginfo)
    exit(-1)
if len(dirlist) != len(nskiplist):
    print("ERROR: incorrect arguments. Each dir should have nskip.")
    print(arginfo)
    exit(-1)


histo = np.zeros(nbins, dtype=float)
rawdata = []  # list of all h-z0 values
assert zmax > zmin
binsize = (zmax - zmin) / nbins
framecounter = 0  # Total number of frames
n_indep = 0  # Number of independent frames

for d, nskip in zip(dirlist, nskiplist):
    if d[0] == "/":
        dd = d
        os.chdir(dd)
    else:
        dd = os.path.join(rootdir, d)
        os.chdir(dd)
    # Parse one directory:
    nbeads = 0  # Number of beads in current dir
    nframes_local = 0  # Number of frames in current dir
    for file in sorted(os.listdir(".")):
        if fnmatch.fnmatch(file, "simulation.pos_*.xyz"):
            print("Parsing %s/%s ..." % (dd, file))
            fdopen = open(file, "r")

            # skip non-thermalized beginning:
            for i in range(nskip):
                read_file_raw("xyz", fdopen)

            # read the first step:
            rr = read_file_raw("xyz", fdopen)
            metainfo = rr["comment"].split()
            for i, word in enumerate(metainfo):
                if word == "Step:":
                    zerostep = int(metainfo[i + 1])
                    step = zerostep
                elif word == "Bead:":
                    beadindex = int(metainfo[i + 1])
            zeronatoms = rr["natoms"]

            nbeads += 1
            histo, rawdata = parse_frame(rr, histo, rawdata, framecounter)
            nframes_local += 1

            while True:
                try:
                    # read the next step:
                    rr = read_file_raw("xyz", fdopen)
                    metainfo = rr["comment"].split()
                    for i, word in enumerate(metainfo):
                        if word == "Step:":
                            step = int(metainfo[i + 1])
                        elif word == "Bead:":
                            if beadindex != int(metainfo[i + 1]):
                                raise RuntimeError(
                                    "The bead index within one file is not constant."
                                )
                        if rr["natoms"] != zeronatoms:
                            raise RuntimeError(
                                "The number of atoms within one file is not constant."
                            )
                    histo, rawdata = parse_frame(rr, histo, rawdata, framecounter)
                except EOFError:
                    break
                framecounter += 1
                nframes_local += 1
    # Finished parsing one directory
    if nbeads not in [
        1,
        6,
        12,
        16,
        24,
    ]:  # I never did different amount of beads so far.
        raise RuntimeError("Weird number of beads (%i) found in dir %s." % (nbeads, dd))
    if nframes_local % nbeads != 0:
        raise RuntimeError("nframes_local % nbeads != 0.\nBroken pos_*.xyz files?")
    # I add 1 because trajectories are independent
    # even if their length is less than 1 corrtime per traj.
    n_indep += 1 + nframes_local // nbeads // corrtime
    os.chdir(rootdir)

print("sum(histo) = %i" % np.sum(histo))

histo = histo / np.sum(histo)
x_axis = np.zeros(nbins, dtype=float)
for i in range(nbins):
    x_axis[i] = i * binsize + zmin

hmx = half_max_x(x_axis, histo)
FWHM = hmx[1] - hmx[0]  # Full width at half maximum
FWHM_middle = (hmx[1] + hmx[0]) / 2
rawdata = np.asarray(rawdata)
average = np.average(rawdata[:, 1])
sigma = np.std(rawdata[:, 1])
error = sigma / np.sqrt(n_indep)
print("FWHM is %f , FWHM middle position at %f A." % (FWHM, FWHM_middle))
print("Mean value is %f, sigma is %f." % (average, sigma))

# Output of a histogram:
np.savetxt(
    "g_z_center-of-mass_%s.dat" % (os.path.split(dirlist[0])[-1]),
    np.c_[x_axis, histo],
    fmt="%.4f",
    header="delta_Z_COM-Rh(111)_[A]    g(z)\n"
    "Data gathered by the following command: %s\n"
    "Total number of frames:  %i\n"
    "FWHM is %f, FWHM middle at %f A,\n"
    "Mean value is %.3f, sigma is %.3f A.\n"
    "Assuming %i steps correlation time, error is %f. N_indep is %i"
    % (
        " ".join(line.strip() for line in sys.argv),
        framecounter,
        FWHM,
        FWHM_middle,
        average,
        sigma,
        corrtime,
        error,
        n_indep,
    ),
)

# Here you can output all distances from frames:
# Because I have rawdata for 2 molecules, I want to reshape and output them in separate columns:
rawdata = rawdata.reshape(-1, 4)
np.savetxt(
    "g_z_center-of-mass_%s.rawdata.dat" % (os.path.split(dirlist[0])[-1]),
    rawdata[:, [0, 1, 3]],
    fmt="%i    %.5f    %.5f",
    header="Data gathered by the following command: %s\n"
    "Total number of frames:  %i\n"
    "Note: you see 2 columns because of 2 molecules in the cell.\n"
    "Mean value is %.3f A, sigma is %.3f A.\n"
    "Assuming %i steps correlation time, error is %f . N_indep is %i"
    % (
        " ".join(line.strip() for line in sys.argv),
        framecounter,
        average,
        sigma,
        corrtime,
        error,
        n_indep,
    ),
)

# Plot histogram:
plt.xlabel('distance to surface (Angstrom)')
plt.grid()
plt.xlim((1., 6.))
plt.plot(x_axis, histo)
plt.show()

print("Done.")
print(
    "ATTENTION:\n\tThis program expects data from cyclohexane PIMD calculations!\n\tFor any other case molecules should be redefined."
)
