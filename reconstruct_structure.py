#!/usr/bin/python3
import sys
import numpy as np

""" This scripts reads an MD trajectory for paracetamol, and reconstructs the molecules as a whole, in case they were broken (whic happens when, e.g., FHI-aims maps all the atoms back to the original unit cell.
    The script uses a reference structure (for which the structure is not broken), and compares the coordinates of each structure to the reference one, atom by atom. 
    If the difference is larger than a given threshold, we add one of the lattice vectors or a combination of them and recheck the newly positioned atom.
"""

filename = sys.argv[1] # trajectory file, xyz format
refname = sys.argv[2]  # reference structure, xyz format

# Define the lattice vectors

# Paracetamol I
a_ = np.array([6.97240019,0,0])
b_ = np.array([0,9.14750004,0])
c_ = np.array([5.18213131,0,11.71621946])
zero = np.array([0,0,0])

# Paracetamol II
#a_=np.array([11.5908,0,0])
#b_=np.array([0,7.3183,0])
#c_=np.array([0,0,17.2258])

# Aspirin I
#a_=np.array([11.41600000,      0.00000000,      0.00000000])
#b_=np.array([0.00000000,      6.59800000,      0.00000000])
#c_=np.array([-1.12050000,      0.00000000,     11.42820000])

# Aspirin II
#a_=np.array([12.357800000,      0.00000000,      0.00000000])
#b_=np.array([0.00000000,      6.531500000,      0.00000000])
#c_=np.array([-4.38760000,      0.00000000,     10.62620000])

print (a_,b_,c_)

def translate(line,shift):
   """
      Shifts one of the atoms by a given vector.
   """
   pos = np.array(list(map(float,line.split()[1:4])))
   atmtyp = line.split()[0]
   pos = pos + shift
   line = atmtyp+'      '+str("{0:.5e}".format(pos[0]))+'  '+str("{0:.5e}".format(pos[1]))+'  '+str("{0:.5e}".format(pos[2]))+'\n'
   return line

# Define possible combinations of lattice vectors
comb = np.array([a_,-a_,b_,-b_,c_,-c_,a_+b_,a_-b_,-a_+b_,-a_-b_,a_+c_,a_-c_,-a_+c_,-a_-c_,b_+c_,b_-c_,-b_+c_,-b_-c_,a_+b_+c_,a_+b_-c_,a_-b_+c_,a_-b_-c_,-a_+b_+c_,-a_+b_-c_,-a_-b_+c_,-a_-b_-c_])

# Read files and store them
with open(filename, 'r') as file0:
  data = file0.readlines()

with open(refname, 'r') as file1:
  ref = file1.readlines()

nat = int(ref[0]) # Number of atoms
print ("There are ",nat, "atoms")
k = 0 # Frame index
counter = 0
nconf = len(data)//(nat+2) # Number of frames (=number of configurations in the file)
print ("There are ", nconf, "configurations in the file")

# Create reference position
posref = []
for i in range(2,nat+2,1): # Go through all the atomic positions
  posref.append(list(map(float,ref[i].split()[1:4])))
posref = np.array(posref)

# Define threshold (will have the same unit as what is given in input )
# Note: the value defined below is really arbitrary, but has worked perfectly for all my systems. Try fiddling with it if the reconstruction doesn't work for your system.
threshold = 3 
while (counter < nconf):
  counter+=1
  for i in range(2,nat+2,1): # Go through all the atomic positions
    pos = np.array(list(map(float,data[k+i].split()[1:4])))
    diff = pos-posref[i-2] # Difference in position between the reference structure and the current one (1 atom)

    # The following assumes that during MD, the positions are all relatively close to each other, in the sense that the amplitude of the atoms is much less than the length of a lattice vector
    if (abs(diff[0]) >threshold or abs(diff[1]) >threshold or abs(diff[2]) >threshold): # Check if the current position is too different from the reference
      for disp in comb: # Go through all the possible combinations of unit-cell parameters
        newpos = pos + disp # Shift the current position
        diff2 = newpos-posref[i-2]
        # Check if the new position is "close enough" to that of the reference one, and if yes, modify the data and exit the current loop
        if (abs(diff2[0]) <threshold and abs(diff2[1]) <threshold and abs(diff2[2]) <threshold):
          data[k+i] = translate(data[k+i],disp)
          break
  
  # Define last structure as reference structure (now the threshold could actually be set much lower, since 2 consecutive MD structures should have very close coordinates...) 
  for i in range(2,nat+2,1): # Go through all the atomic positions
    pos = np.array(list(map(float,data[k+i].split()[1:4])))
    posref[i-2] = pos

  # Go to the next configuration (each one contains nat+2 lines)
  k+=nat+2 

# Finally write everything back to another file
with open('reconstructed_'+filename, 'w') as file2:
    file2.writelines( data )
