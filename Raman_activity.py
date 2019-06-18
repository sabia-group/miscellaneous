#!/usr/bin/python3
import sys
import numpy as np

"""
   Reads an xyz file containing vibrational modes (as outputted by FHI-aims), and tells whether they are Raman-active or not.
   For now, the script is not fully general. Indeed, one needs to know beforehands the symmetry group of the system under study (not a problem), and write out the corresponding symmetry operations. I only did it for C2h (Paracetamol I, Aspirin I and II) and D2h (Paracetamol II).
   Also, I *think* the coordinates must be centered around the origin and the structure not be broken, although I would need to check this.
   The script will find the mapping between atoms for each symmetry operation.
   Then it will apply these symmetry operations to the eigenmodes, and determine the character table for each mode.
   A way to make it way more general would be to use spglib: the latter would determine the spacegroup of the system and the mapping. Once the mapping is obtained and the symmetry operations known, the end of the script would remain identical to what it is now.
"""

def retrieve_coords(filename):
  """
     Reads an xyz file and returns the number of atoms, a list of atom types and a list of positions.
  """
  attyp = []
  pos = []
  data = open(filename,'r')
  line = data.readline()
  line2 = line.split()
  nat = int(line2[0])
  line = data.readline()
  for i in range(nat):
    line = data.readline()
    line2 = line.split()
    attyp.append(line2[0])
    pos.append(list(map(float,line2[1:4])))
  data.close()
  pos = np.array(pos)
  return nat,attyp,pos
      
        
def find_mapping(attyp,pos,matrix,trans,comb):
  """
     Reads in a list of atoms (list of atom types and list of atom positions), as well as a rotation matrix, a translation vector, and a list of possible combination of lattice vectors.
     It returns the mapping of atoms for te current symmetry operation in the form of a list of indexes, indicating the correspondence with the original atoms.
  """

  # Rotate and translate the structure
  #pos_sym = np.dot(pos,matrix) + trans
  pos_sym = pos@matrix + trans

  # Compare element by element the 2 arrays, eventually adding a translation vector to coords_sym if both coords are not equal
  index = []
  for i,el in enumerate(pos):
    for i_sym,el_sym in enumerate(pos_sym):
      if (attyp[i]==attyp[i_sym]): # Compare only elements of the same type

  # Loop over the possible unit cell translations: some atoms are indeed mapped onto atoms belonging to a neighbouring cell, so we need to displace them by of of te unit vector combination and check the position in each case to find a match
        for disp in comb:
          new_el_sym = el_sym + disp
          diff = abs(new_el_sym - el)

 # When a match is found (i.e., when the difference in position between the original and rotated/translated structure is lower than a given threshold), add the index of the original array the new array corresponds to
          if ((diff<0.1).all()):
            index.append(i_sym)
            break

  if (len(index) < len(pos)): # Every atom should be mapped
    print ("Error: The system does not seem to have the indicated symmetry")
    sys.exit()

  return index



filename = sys.argv[1] # Name of input file

# Retrieve number of atoms, as well as the elements and positions
nat,attyp,coords = retrieve_coords(filename)

# Define the lattice vectors
# This is hard-coded but could be read from file, if the information is present

# Paracetamol I old lattice
a_=np.array([6.97240019,0,0])
b_=np.array([0,9.14750004,0])
c_=np.array([5.18213131,0,11.71621946])

# Paracetamol I old lattice supercell
#a_=np.array([13.94480038,0,0])
#b_=np.array([0,9.14750004,0])
#c_=np.array([5.18213131,0,11.71621946])

# Paracetamol I correct lattice
#a_=np.array([7.08,0,0])
#b_=np.array([0,9.34,0])
#c_=np.array([5.532,0,11.598])

# Paracetamol II (old lattice)
#a_=np.array([11.5908,0,0])
#b_=np.array([0,7.3183,0])
#c_=np.array([0,0,17.2258])

# Paracetamol II (correct lattice)
#a_=np.array([11.84,0,0])
#b_=np.array([0,7.40,0])
#c_=np.array([0,0,17.16])

# Aspirin I exp lattice 300K
#a_=np.array([11.41600000,      0.00000000,      0.00000000])
#b_=np.array([0.00000000,      6.59800000,      0.00000000])
#c_=np.array([-1.12050000,      0.00000000,     11.42820000])

# Aspirin I exp lattice 123K
#a_=np.array([11.277600000,      0.00000000,      0.00000000])
#b_=np.array([0.00000000,      6.506400000,      0.00000000])
#c_=np.array([-1.14690000,      0.00000000,     11.21910000])

# Aspirin I calc lattice
#a_=np.array([11.503100000,      0.00000000,      0.00000000])
#b_=np.array([0.00000000,      6.506900000,      0.00000000])
#c_=np.array([-1.23056000,      0.00000000,     11.389920000])

# Aspirin II exp lattice
#a_=np.array([12.35780000,  0.00000000, 0.00000000])
#b_=np.array([ 0.00000000, 6.53150000, 0.00000000])
#c_=np.array([ -4.38760000,  0.00000000, 10.62620000])

# Aspirin II calc lattice
#a_=np.array([12.52030000,  0.00000000, 0.00000000])
#b_=np.array([ 0.00000000, 6.64970000, 0.00000000])
#c_=np.array([ -4.29273471,  0.00000000, 11.01637358])


# Combination of lattice vectors
zero = np.array([0,0,0])
comb = np.array([zero,a_,-a_,b_,-b_,c_,-c_,a_+b_,a_-b_,-a_+b_,-a_-b_,a_+c_,a_-c_,-a_+c_,-a_-c_,b_+c_,b_-c_,-b_+c_,-b_-c_,a_+b_+c_,a_+b_-c_,a_-b_+c_,a_-b_-c_,-a_+b_+c_,-a_+b_-c_,-a_-b_+c_,-a_-b_-c_])

list_matrix = []
list_trans = []
list_index = []

# The following symmetry operations (and character tables) are hard-coded. There is no other way around it, though. However, it has been done before, so it would be best using existing softwares/libraries for that, like spglib.
# Here I only did it for C2h and D2h. Also, it would be better to convert these to dictionaries and place them in an auxiliary file.

# I) Valid for C2h, i.e., aspirin I and II and paracetamol I

#Sym op 1: C2y
matrix = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
trans = b_/2 + c_/2
list_matrix.append(matrix)
list_trans.append(trans)
index = find_mapping(attyp,coords,matrix,trans,comb)
list_index.append(index)

#Sym op 2: Inversion center
matrix = np.array([[-1,0,0],[0,-1,0],[0,0,-1]]) 
trans = np.array([0,0,0])
list_matrix.append(matrix)
list_trans.append(trans)
index = find_mapping(attyp,coords,matrix,trans,comb)
list_index.append(index)

#Sym op 3:  sigma_xz
matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]])
trans = b_/2 + c_/2
list_matrix.append(matrix)
list_trans.append(trans)
index = find_mapping(attyp,coords,matrix,trans,comb)
list_index.append(index)

Ag = ['Ag',['+1','+1','+1','+1'],'active']
Bg = ['Bg',['+1','-1','+1','-1'],'active']
Au = ['Au',['+1','+1','-1','-1'],'inactive']
Bu = ['Bu',['+1','-1','-1','+1'],'inactive']
char_ref = [Ag,Bg,Au,Bu]


# II) Valid for D2h, i.e., paracetamol II

##Sym op 1: C2z
#matrix = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) 
#trans = a_/2 + c_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 2: C2y
#matrix = np.array([[-1,0,0],[0,1,0],[0,0,-1]]) 
#trans = b_/2 + c_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 3: C2x
#matrix = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) 
#trans = a_/2 + b_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 4: inversion
#matrix = np.array([[-1,0,0],[0,-1,0],[0,0,-1]]) 
#trans = np.array([0,0,0])
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 5: sigma_xy
#matrix = np.array([[1,0,0],[0,1,0],[0,0,-1]]) 
#trans = a_/2 + c_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 6: sigma_xz
#matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]]) 
#trans = b_/2 + c_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
##Sym op 7: sigma_yz
#matrix = np.array([[-1,0,0],[0,1,0],[0,0,1]]) 
#trans = a_/2 + b_/2
#list_matrix.append(matrix)
#list_trans.append(trans)
#index = find_mapping(attyp,coords,matrix,trans,comb)
#list_index.append(index)
#
#
#Ag= ['Ag ',['+1','+1','+1','+1','+1','+1','+1','+1'],'active']
#B1g=['B1g',['+1','+1','-1','-1','+1','+1','-1','-1'],'active']
#B2g=['B2g',['+1','-1','+1','-1','+1','-1','+1','-1'],'active']
#B3g=['B3g',['+1','-1','-1','+1','+1','-1','-1','+1'],'active']
#Au= ['Au ',['+1','+1','+1','+1','-1','-1','-1','-1'],'inactive']
#B1u=['B1u',['+1','+1','-1','-1','-1','-1','+1','+1'],'inactive']
#B2u=['B2u',['+1','-1','+1','-1','-1','+1','-1','+1'],'inactive']
#B3u=['B3u',['+1','-1','-1','+1','-1','+1','+1','-1'],'inactive']
#char_ref=[Ag,B1g,B2g,B3g,Au,B1u,B2u,B3u]


# Now that we know the mapping, determine character table for eigenmodes

data = open(filename,'r') # xyz format

while(True):
  eigen = []
  line = data.readline()
  if not line:
    break
  line2 = line.split()
  nat = int(line2[0])
  line = data.readline() # Comment line
  #print(line.split('\n')[0]) # Reprint the line as it is
  freq = line.split()[3] # Frequency of the mode
  # Retrieve eigenvectors
  for i in range(nat):
    line = data.readline()
    line2 = line.split()
    eigen.append(list(map(float,line2[4:7])))
  eigen = np.array(eigen)

  char_table = ['+1'] # The first character is always +1 (identity)
  found = False # tells if we have found a match for the vibrational mode

  warning = ''
  for matrix,trans,index_2 in zip(list_matrix,list_trans,list_index):
    # Rotate eigenvectors (Note that no translation is applied here, contrary to the positions...)
    eigen_sym = eigen@matrix 


    # Apply symmetries on the total eigenvector (nat*3 array) and check the sign of the dot product. Doing so, we would simply miss an anomaly for a given atom if there is any, but in principles there is none (except for the negative frequency modes, which are anyway non physical) if the calculations are reasonably converged.

    # Reorder the eigenvector that underwent a symmetry operation so that it matches the original eigenvector
    eigen_sym_reord = [] 
    for i_sym in index_2:
      eigen_sym_reord.append(eigen_sym[i_sym])
    prod = np.sum(eigen*eigen_sym_reord)

    # Determine characters:
    if prod > 0:
      char_table.append('+1')
    else:
      char_table.append('-1')

    if ( abs(np.linalg.norm(eigen)-np.sqrt(abs(prod))) > 5e-3 ):
      warning = "This mode is not perfectly symmetric"

  # Now compare the character of the current mode to the reference character table
  for ch in char_ref:
    if (char_table == ch[1]):
      found = True
      print(freq, char_table, ch[0], ch[2], warning)
      break

  if (found == False):
    print(freq, char_table, 'This is not a proper vibrational mode')
      

