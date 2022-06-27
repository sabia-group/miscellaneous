#!/usr/bin/env python3
""" This program collects the data from single point calculations from PIMD trajectories.
    It outputs all work function values together with corresponding c.o.m. distance to surface
    and a histogram of the work function.
    It puts an error estimation to the headers of the output files,
    but please adjust the 'corrtime' variable for your system.
"""
# If you encounter a problem that ASE does not recognize D species, try the following:
# In .local/lib/python3.5/site-packages/ase/io/aims.py:50 should be this code:
#        if inp[0] == "atom":
#            cart_positions = True
#            if xyz.all():
#                fix.append(i)
#            elif xyz.any():
#                fix_cart.append(FixCartesian(i, xyz))
#            floatvect = float(inp[1]), float(inp[2]), float(inp[3])
#            positions.append(floatvect)
#            magmoms.append(0.0)
#            charges.append(0.0)
#            symbols.append(inp[-1])
#            if symbols[-1] == 'D':
#                symbols[-1] = 'H'
#                warn("Custom read function substitutes D by H!", UserWarning)

import sys, os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import re
from ipi.utils.units import unit_to_internal, unit_to_user
from ase.io import read

rootdir = os.getcwd()
dirlist = None
wf = []
dist = []
corrtime = 600  # Assuming 300 fs corrtime (600 steps typically)


def read_aims_workf(file):
#   Reads the file and returns the last value of work function,
#   but only if 'Have a nice day' is found.
    wf = 0.  # Added this line to use this parser for isolated molecules also. Otherwise it breaks on files w/o work function.

    fdopen = open(file, 'r')
    for line in fdopen.readlines():
        if "| Work function (\"upper\" slab surface)" in line:
            wf = float(re.search('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', line).group())
        if line == "          Have a nice day.\n":
            fdopen.close()
            if wf == 0:
                print((file, "    wf = 0"))
            return wf
    fdopen.close()
    return None


def get_distance(atoms):
    """ Here it doesn't make sense to separate molecules, since wf value is common anyways
    """
    zmol = []
    zsurf = []
    mmol = []
    for atom in atoms:
        if atom.index in range(0,36):
            if atom.symbol not in ('C','H','D'):
                raise RuntimeError("Why atom %s in first 36 atoms?" % atom.symbol)

            zmol.append(atom.position[2])
            mmol.append(atom.mass)
        elif atom.symbol == 'Rh':
            zsurf.append(atom.position[2])
    zmol = np.asarray(zmol)
    zsurf = np.asarray(zsurf)
    mmol = np.asarray(mmol)
    com_mol = np.dot(mmol, zmol) / mmol.sum()
    avg_surf = np.average(np.sort(zsurf)[-25:])

    d = com_mol - avg_surf
    if(d < 0):
        raise RuntimeError("The center of mass of molecules is below the surface?")
    return d

def parseOneDir(wf, dist):
    """ Parses PIMD directory where it called from.
        Searches for subfolders of type step_*/bead_*
    """
    cwd = os.getcwd()
    nlocal = 0
    for dir1 in sorted(fnmatch.filter(os.listdir('.'), 'step_?????')):
        print(dir1)
        nbeads = 0
        nsublocal = 0
        for dir2 in sorted(fnmatch.filter(os.listdir(dir1), 'bead_??')):
            nbeads += 1
            dir = os.path.join(cwd, dir1, dir2)
            if 'aims.out' in os.listdir(dir):
                file = os.path.join(dir, 'aims.out')
                a = read_aims_workf(file)
                if (a != None):
                    wf.append(a)
                atoms = read(os.path.join(dir, 'geometry.in'), format='aims')
                d = get_distance(atoms)
                dist.append(d)
                nsublocal += 1
        if nsublocal != nbeads and nsublocal != 0:
            raise RuntimeError("folder %s is incomplete." % cwd)
        elif nsublocal == 0:
            # Means that this folder is prepared,
            # but has not been touched by slurm-loop.sh yet
            pass
        else:
            nlocal += 1
    return nlocal


if __name__ == "__main__":
    nframes = 0     # counter for single frames
    n_indep = 0     # counter for independent snapshots
    infotext = ("\n\'parse_wf_calculations_v4.1\' outputs all w.f. values "
        "together with the corresponding c.o.m. distance to the surface,"
        "\nplus a histogram of the work function."
        "\nDon't use it for classical calculations - "
        "they should be parsed in cyclohexane/pimd/ directly."
        "\nPossible arguments:\n"
        "\t1:\t'here' - for parsing current directory only\n"
        "\t>=2:\t-d dir1 dir2 ... - for parsing given directories "
        "to a single resulting file.\n"
        "Keep in mind that 'dir1' will be used for naming the output file.")
    if (len(sys.argv) == 1):
        print(infotext)
        exit(-1)
    if (len(sys.argv) == 2):
        if sys.argv[1] == 'here':
            dirlist = [os.getcwd()]
        else:
            print(infotext)
            exit(-1)
    elif sys.argv[1] == '-d':
        dirlist = sys.argv[2:]
    else:
        print(infotext)
        exit(-1)

    if dirlist is None:
        print("ERROR: directories expected.")
        print(infotext)
        exit(-1)

    # parsing multiple directories:
    for d in dirlist:
        if d[0] == '/':
            dd = d
        else:
            dd = os.path.join(rootdir, d)
        os.chdir(dd)
        nlocal = parseOneDir(wf, dist)
        nframes += nlocal
        # Assuming interval of 60 steps in 'prepare_wf_calculations.py'
        n_indep += 1 + nlocal // (corrtime / 60)  # I add 1 because trajectories are always independent
        os.chdir(rootdir)

    # Analyzing the gathered data:
    wf = np.asarray(wf)
    wf_avg = np.average(wf)
    wf_stdev = np.std(wf)
    wf_error = wf_stdev / np.sqrt(n_indep)

    dist = np.asarray(dist)
    dist_avg = np.average(dist)
    dist_stdev = np.std(dist)
    dist_error = dist_stdev / np.sqrt(n_indep)

    print("Number of entries: %i" % wf.size)
    print("%40s\t%16f\n%40s\t%16f\n%40s\t%16f\n"
          % ('Average work function [eV]:',
             wf_avg,
             'sigma:',
             wf_stdev,
             'error:',
             wf_error)
         )
    print("Average distance from monolayer's c.o.m. to surface:")
    print("%40s\t%16f\n%40s\t%16f\n%40s\t%16f\n"
          % ("Average work function [eV]:",
             dist_avg,
             "sigma:",
             dist_stdev,
             "error:",
             dist_error)
         )

    print("\nWriting all work function values and "
          "distances from c.o.m. to surface to file")

    # Output w.f. to a file:
    np.savetxt('dist_and_wf-values_%s.dat'% os.path.split(dirlist[0])[-1],
               np.column_stack((dist, wf)),
               header="Distances from monolayer's c.o.m. to surface and work function values "
                      "gathered from the following directories: %s\n"
                      "Mean_distance %f,  mean wf %f\n"
                      "Errors: %g,  %g. N_indep: %i"
                      % (", ".join(line.strip() for line in dirlist),
                         dist_avg,
                         wf_avg,
                         dist_error,
                         wf_error,
                         n_indep
                        )
              )

    # building a histogram
    xmin, xmax = (3.8, 4.6)
    nbins = 160
    print(
        "\nWork function histogram settings: range [%g:%g], %i bins, normalized"
        % (xmin, xmax, nbins))
    histo, bin_edges = np.histogram(wf,
                                    bins=nbins,
                                    range=(xmin, xmax),
                                    density=True)
    # output w.f. histogram to a file:
#    print("Number of entries: %i" % wf.size)
#    print("%40s\t%16f\n%40s\t%16f\n"
#          % ('Average work function [eV]:', wf_avg, 'sigma / sqrt(n):', wf_error))
    np.savetxt(
        'wf-histo_%s.dat'% os.path.split(dirlist[0])[-1],
        np.c_[bin_edges[:-1], histo],
        header=(
            "Work function gathered from the following directories: %s\n"
            "Mean w.f. [eV]: %.5g    sigma: %g    error: %g\n"
            "Number of entries:  %i;    N_indep: %i.\n"
            % (", ".join(line.strip() for line in dirlist),
               wf_avg,
               wf_stdev,
               wf_error,
               wf.size,
               n_indep
              )
        )
    )
