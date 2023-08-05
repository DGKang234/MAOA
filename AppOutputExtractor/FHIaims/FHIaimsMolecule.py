from NonPeriodic.Molecule import BaseMolecule
from NonPeriodic.Atom import BaseAtom

import os,re
import numpy as np

'''

'''

class atom(BaseAtom):

	def __init__(self,atom_type='atom',atom_attr='default',x=0.,y=0.,z=0.):

		'''
		
		'''

		super().__init__(atom_type=atom_type,atom_attr=atom_attr,x=x,y=y,z=z)


class molecule(BaseMolecule):

	def __init__(self,geometry_file):

		'''

		'''

		super().__init__()	# BaseMolecule NonPeriodic/ (number_of_atoms,atom_list)

		try:
			with open(geometry_file,'r') as f:
				self.file_exist = True
				self.file_path  = geometry_file
				pattern = re.compile(r"atom")

				for line in f:
					if pattern.search(line):
						ls = line.split()
						new_atom = atom(ls[4],ls[0],ls[1],ls[2],ls[3])	
						self.add_atom(new_atom)
	
		except FileNotFoundError as e:
			self.file_exist = False
			print(e)	# pass

	def is_exist(self):
		return self.file_exist


'''

'''

def calculate_rmsd_molecules(moleculeA,moleculeB):

	if moleculeA.get_number_of_atoms() != moleculeB.get_number_of_atoms():
		return False
	else:
		rmsd = 0. 

		for atomA,atomB in zip(moleculeA.get_atomlist(),moleculeB.get_atomlist()):
			cartA = np.array(atomA.get_cart())
			cartB = np.array(atomB.get_cart())
			dev = np.linalg.norm(cartA - cartB)
			rmsd = rmsd + dev		

		try:
			rmsd = rmsd/(float(moleculeA.get_number_of_atoms())*3.)
			return rmsd
		except ZeroDivisionError as e:
			print(e)
			return None


'''

'''

if __name__ == '__main__':

	print('Molecule - 1')
	fmol = molecule('/Users/woongkyujee/Desktop/Python/AppOutputAnalysis/unit_tests/run_1/geometry.in')
	print(fmol.is_exist())
	fmol.show_info()

	print('Molecule - 2')
	fmol_2 = molecule('/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_2/geometry.in')
	fmol_2.show_info()
	#fmol_2.get_atom(2).show_info()

	print('rmsd test 1')
	print(calculate_rmsd_molecules(fmol,fmol_2))

	print('rmsd test 2')
	fmolA = molecule('geoA.txt')
	fmolB = molecule('geoB.txt')
	print(fmolA.show_info())
	print(fmolB.show_info())
	print(calculate_rmsd_molecules(fmolA,fmolB))
