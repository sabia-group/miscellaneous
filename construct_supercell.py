#!/usr/bin/python3
import sys
import numpy as np

"""
   This scripts reads an MD trajectory (xyz format) and converts each geometry into a supercell geometry of size 2x2x2.
"""

filename=sys.argv[1] # xyz format

# Define the lattice vectors

uc = np.array([0,0,0])

# Paracetamol I
a_=np.array([6.97240019,0,0])
b_=np.array([0,9.14750004,0])
c_=np.array([5.18213131,0,11.71621946])

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



comb = np.array([uc,a_,b_,a_+b_,c_,c_+a_,c_+b_,c_+a_+b_]) # For a 2x2x2 supercell

f_ = open(filename,'r')

while(True):
  pos = []
  attyp = []
  line = f_.readline()
  if not line:
    break
  nat = int(line.split()[0])
  linecomm = f_.readline()
  for i in range(nat):
    line = f_.readline()
    pos.append(list(map(float,line.split()[1:4])))
    attyp.append(line.split()[0])
  print(8*nat)
  print (linecomm.split('\n')[0])
  # Go through all vector dislacements and print coordinates
  for disp in comb:
    newpos = pos+disp
    for i,el in enumerate(newpos):
      print (attyp[i],el[0],el[1],el[2])

