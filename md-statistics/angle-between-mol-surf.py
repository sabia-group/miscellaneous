#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This program takes 'simulation.pos_*.xyz' files, and calculates
    the distribution of angles between surface and cyclohexane molecules.
    Can parse several directories at once.
"""

import sys, os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from ipi.utils.io import read_file_raw
from sklearn.decomposition import PCA


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def parse_frame(rr, histo, rawdata, framecounter):
    """Parses single frame and adds found atoms to the histogram provided as an argument.
    arguments:
        rr      - frame read by read_file_raw()
        histo   - histogram to add points to
        species - which species to search
    """
    surface = []
    mol1 = []
    hydrogens = []
    for i in range(rr["natoms"]):
        if rr["names"][i] == "C":
            mol1.append(rr["data"][3 * i : 3 * i + 3])
        if rr["names"][i] == "Rh":
            surface.append(rr["data"][3 * i + 2])
        elif rr["names"][i] in ("H", "D"):
            hydrogens.append(rr["data"][3 * i + 2])

    if len(surface) == 0:
        raise RuntimeError("Rhodium not found")
    if len(hydrogens) != 12:
        raise RuntimeError("Why there is not 12 hydrogens?")

    surface = sorted(surface, key=float)
    topsurface = np.asarray(surface[-9:])
    z0 = np.average(topsurface)

    for h in hydrogens:
        d = h - z0
        if d < 0:
            raise RuntimeError("Some atoms are below the surface.")

    data = np.asarray(mol1)
    # Principal Component Analysis
    pca = PCA(n_components=2)
    pca.fit(data)
    normale = np.cross(pca.components_[0], pca.components_[1])
    # Make sure the normale always points to the top
    if normale[2] < 0:
        normale = -normale
    angle_deg = 180 / np.pi * angle_between(normale, [0.0, 0.0, 1.0])
    rawdata.append([framecounter, angle_deg])

    histo[int(np.floor(angle_deg / binsize))] += 1

    return histo, rawdata


# ======================== The algorithm starts here ========================
rootdir = os.getcwd()
dirlist = []
zerostep = 0
zeronatoms = 0  # number of atoms in the first read frame
step = -1

nskip = None
species = None

nbins = 360
angmin = 0.0
angmax = 90.0

print(
    "'angle-mol-surf.py' takes 'simulation.pos_*.xyz' files, and calculates\n"
    "the distribution of angles between surface and cyclohexane molecules.\n"
    "It can parse several directories at once.\n"
)

if len(sys.argv) == 1:
    print(
        "Possibe arguments:\n"
        "\t-nskip <nskip> - number of frames to skip in the beginning of the file\n"
        "\teither '-d' or '-here':\n"
        "\t\t-d <dir1> <dir2> ... <dirN> - directories to take trajectories from\n"
        "\t\t-here to use current directory\n"
        "\t\tNote that dir1 will be used for naming output files.\n"
        "optionally:\n"
        "\t-nbins <nbins> - number of bins for histogram  [%i]\n"
        "\t-angmin <angmin> - minimal angle for the histogram [%g]\n"
        "\t-angmax <angmax> - maximal angle for the histogram [%g degrees]\n"
        % (nbins, angmin, angmax)
    )
    exit(-1)
else:
    for i, arg in enumerate(sys.argv):
        if arg == "-nskip":
            nskip = int(sys.argv[i + 1])
            assert nskip > 0
        if arg == "-d":
            for j in range(i + 1, len(sys.argv)):
                if sys.argv[j][0] != "-":
                    dirlist.append(sys.argv[j])
                else:
                    break
        if arg == "-here":
            dirlist.append(rootdir)
        if arg == "-nbins":
            nbins = int(sys.argv[i + 1])
        if arg == "-angmin":
            angmin = float(sys.argv[i + 1])
        if arg == "-angmax":
            angmax = float(sys.argv[i + 1])
if "-d" in sys.argv and "-here" in sys.argv:
    print("ERROR:\t\t-d and -here keys are incompatible!")
    exit(-1)

if nskip is None or dirlist == []:
    print(
        "ERROR: incorrect arguments.\nMandatory fields:\n"
        "\t-nskip <nskip> - number of frames to skip in the beginning of the file\n"
        "\teither '-d' or '-here':\n"
        "\t\t-d <dir1> <dir2> ... <dirN> - directories to take trajectories from\n"
        "\t\t-here to use current directory\n"
        "\t\tNote that dir1 will be used for naming output files.\n"
        "Optional:\n"
        "\t-nbins <nbins> - number of bins for histogram  [%i]\n"
        "\t-angmin <angmin> - minimal angle for the histogram [%g]\n"
        "\t-angmax <angmax> - maximal angle for the histogram [%g degrees]\n"
        % (nbins, angmin, angmax)
    )
    exit(-1)

histo = np.zeros(nbins, dtype=float)
rawdata = []  # list of all angle values
assert angmax > angmin
binsize = (angmax - angmin) / nbins
framecounter = 0

for d in dirlist:
    if d[0] == "/":
        dd = d
        os.chdir(dd)
    else:
        dd = os.path.join(rootdir, d)
        os.chdir(dd)
    # Parse one directory:
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

            histo, rawdata = parse_frame(rr, histo, rawdata, framecounter)

            while True:
                try:
                    # read the next step:
                    framecounter += 1
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
    # Finished parsing one dyrectory
    os.chdir(rootdir)

histo = histo / np.sum(histo)
x_axis = np.zeros(nbins, dtype=float)
for i in range(nbins):
    x_axis[i] = i * binsize

rawdata = np.asarray(rawdata)
average = np.average(rawdata[:, 1])
sigma = np.std(rawdata[:, 1])
np.savetxt(
    "angles_%s.histo.dat" % (os.path.split(dirlist[0])[-1]),
    np.c_[x_axis, histo],
    fmt="%8.4g",
    delimiter="    ",
    header="Angle between cyclohexane and Rh(111) surface # data gathered from the following directories: %s # Total number of frames:  %i\n"
    "Mean value is %.3f, sigma is %.3f."
    % (", ".join(line.strip() for line in dirlist), framecounter, average, sigma),
    comments="# ",
)

# Here you can output all angles from frames:
np.savetxt(
    "angles_%s.rawdata.dat" % (os.path.split(dirlist[0])[-1]),
    rawdata,
    fmt="%i  %.5f",
    header="# data gathered from the following directories: %s # Total number of frames:  %i\n"
    % (", ".join(line.strip() for line in dirlist), framecounter),
)

# Plot histogram:
plt.xlabel("Angle between the surface and the molecule.")
plt.grid()
plt.xlim((0.0, 90.0))
plt.plot(x_axis, histo)
plt.show()
