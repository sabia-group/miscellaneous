#! /usr/bin/env python3
""" (C) Karen Fidanyan 2019
    This script takes geometry file for C6H12 on Rh surface and builds E(z)
    by moving molecules and relaxing them with fixed center of mass.
"""
import os, sys
from socket import gethostname
from warnings import warn
import numpy as np
from ase.calculators.aims import Aims
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
# June 2022 note: In my own version of ASE, I added the FixGroupCom() constraint.
# If it never gets into ase/master, look at the commented class here.
# It might be possible to just plug it in instead of the ASE implementation,
# but things may have changed since then.
from ase.constraints import *

# Environment-dependent parameters -- please configure according to machine
species_dir = '/u/kfidan/soft/fhi-aims.dev/species_defaults/light'
command = 'srun -N 4 -n 128 /u/kfidan/bin/aims.190214.scalapack.mpi.x'
hostname = gethostname()
port = 31415

# This class is commented out here to make sure
# that it works in the same way as other constraints.
# class FixGroupCom(FixConstraint):
#    """Constraint class for fixing the center of mass of a subgroup,
#       enlisted in the same way as FixAtoms class does it.
#
#       This class lies also in the '~/.local/lib/python?.?/site-packages/ase/constraints.py',
#       But it should be kept here in case something happens with that file (reinstallation or so)
#    """
#
#    def __init__(self, indices=None, mask=None):
#        """Constrain COM of chosen atoms.
#
#        Parameters
#        ----------
#        indices : list of int
#           Indices for those atoms that should be constrained.
#        mask : list of bool
#           One boolean per atom indicating if the atom should be
#           constrained or not.
#
#        Examples
#        --------
#        Fix all Copper atoms:
#
#        >>> mask = [s == 'Cu' for s in atoms.get_chemical_symbols()]
#        >>> c = FixAtoms(mask=mask)
#        >>> atoms.set_constraint(c)
#
#        Fix all atoms with z-coordinate less than 1.0 Angstrom:
#
#        >>> c = FixAtoms(mask=atoms.positions[:, 2] < 1.0)
#        >>> atoms.set_constraint(c)
#        """
#
#        if indices is None and mask is None:
#            raise ValueError('Use "indices" or "mask".')
#        if indices is not None and mask is not None:
#            raise ValueError('Use only one of "indices" and "mask".')
#
#        if mask is not None:
#            indices = np.arange(len(mask))[np.asarray(mask, bool)]
#        else:
#            # Check for duplicates:
#            srt = np.sort(indices)
#            if (np.diff(srt) == 0).any():
#                raise ValueError(
#                    'FixGroupCom: The indices array contained duplicates. '
#                    'Perhaps you wanted to specify a mask instead, but '
#                    'forgot the mask= keyword.')
#        self.index = np.asarray(indices, int)
#
#        if self.index.ndim != 1:
#            raise ValueError('Wrong argument to FixGroupCom class!')
#
#        self.removed_dof = 3
#
#    def adjust_positions(self, atoms, new):
#        m = atoms.get_masses()[self.index]
#        warn("Center of mass will be incorrect with scaled coordinates.", Warning)
#        # calculate center of mass:
#        old_cm = np.dot(m, atoms.arrays['positions'][self.index,:]) / m.sum()
#        new_cm = np.dot(m, new[self.index,:]) / m.sum()
#        d = np.empty_like(new)
#        d[self.index,:] = old_cm - new_cm
#        new[self.index,:] += d[self.index,:]
#
#    def adjust_forces(self, atoms, forces):
#        m = atoms.get_masses()[self.index]
#        mm = np.tile(m, (3, 1)).T
#        lb = np.empty_like(forces)
#        lb[self.index,:] = np.sum(mm * forces[self.index,:], axis=0) / sum(m**2)
#        mfull = atoms.get_masses()
#        mmfull = np.tile(mfull, (3, 1)).T
#        forces[self.index,:] -= (mmfull * lb)[self.index,:]
#
#    def get_indices(self):
#        return self.index
#
#    def __repr__(self):
#        return 'FixGroupCom(indices=%s)' % ints2string(self.index)
#
#    def todict(self):
#        return {'name': 'FixGroupCom',
#                'kwargs': {'indices': self.index}}


def Optimize_And_Print(atoms=None, energies=None, n=None):
    """ A function to use at each displacement.
        Performs geometry optimization and output of geometries.
        Also appends energy of this displacement
    """
    if atoms is None or energies is None or n is None:
        raise RuntimeError("All arguments of 'Optimize_And_Print()' are mandatory.")

    print(n)
    opt = BFGS(atoms, trajectory='opt.aims.traj', logfile='opt.aims.log')
    opt.run(fmax=0.01, steps=60)
    energies.append(atoms.get_potential_energy(force_consistent=False, apply_constraint=True))

    # reading in the trajectory file created during optimization
    traj = Trajectory("opt.aims.traj", 'r')
    nsteps = opt.get_number_of_steps()

    outFileName_geop = "relaxation_%i.xyz" % n
    if os.path.exists(outFileName_geop):
        os.remove(outFileName_geop)
    # write each structure from the .traj file in .xyz format
    if nsteps == 0:
        write(outFileName_geop, atoms, format='xyz', append=True)
    else:
        for i in range(-nsteps, 0):
            atoms2 = traj[i]
            write(outFileName_geop, atoms2, format='xyz', append=True)

    atoms2 = traj[-1]
    write(outFileName, atoms2, format='xyz', append=True)


# === HERE THE ALGORITHM STARTS ===
energies = []

atoms = read('geometry.initial.in', format='aims')
global outFileName
outFileName = 'relaxed.xyz'

aims = Aims(command=command,
            species_dir=species_dir,
            use_pimd_wrapper=(hostname, port),
            compute_forces=True,
            xc='PBE',
            spin='none',
            relativistic='atomic_zora scalar',
            k_grid=[2, 2, 1],
            # never re-initialize Pulay mixer:
            sc_iter_limit='200',
            sc_init_iter='200',
            # small mixing needed for metal
            charge_mix_param='0.02',
            # big blur also needed for metal
            occupation_type='gaussian 0.1',
            sc_accuracy_forces='1E-3',
            use_dipole_correction=True,
            vdw_correction_hirshfeld=True,
            vdw_pair_ignore='Rh Rh',
            output_level='MD_light')

calc = SocketIOCalculator(aims, log=sys.stdout, port=port)
atoms.calc = calc

warn("Using several FixGroupCom constraints, avoid overlaps of constrained groups, "
     "because it would produce undefined and incorrect behavior.", Warning)
constraints = []  # the list for all constraints applied to a system
constraints.append(FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Rh']))
atoms.set_constraint(constraints)

# For all vdW models except PBE+vdW_surf original geometry is not optimal,
# so I relax it with fixed Rhodium before starting doing energy profile.
print("Performing initial optimization without constraints on molecules (Rh is fixed).")
Optimize_And_Print(atoms, energies, 0)  # n = 0

constraints.append(FixGroupCom(indices=range(0, 6)+range(12, 24)))  # that's how atoms are organized in my geometry.initial.in
constraints.append(FixGroupCom(indices=range(6, 12)+range(24, 36))) # (2 cyclohexane molecules)
atoms.set_constraint(constraints)

print('\nConstraints:')
for i in atoms._constraints: print(i)

# I want to start both slopes of E(z) from the initial position, it may save some geop iterations.
initial_pos = atoms.positions.copy()

# The first branch goes towards the surface
for n in range(1, 11):
    for atom in atoms:
        if atom.symbol in ('C', 'H'):
            atom.position[2] -= 0.05
    Optimize_And_Print(atoms, energies, n)

# The second branch goes from the initial point towards vacuum
atoms.set_positions(initial_pos, apply_constraint=False)

atoms.calc = calc
for n in range(11, 21):
    for atom in atoms:
        if atom.symbol in ('C', 'H'):
            atom.position[2] += 0.05
    Optimize_And_Print(atoms, energies, n)

for n in range(21, 31):
    for atom in atoms:
        if atom.symbol in ('C', 'H'):
            atom.position[2] += 0.1
    Optimize_And_Print(atoms, energies, n)

print('\nEnergies:')
print(energies)
# outEFile = open('Energies.dat', 'w')
# for e in energies:
#     outEFile.write('%12f\n' % e)
# outEFile.close()
