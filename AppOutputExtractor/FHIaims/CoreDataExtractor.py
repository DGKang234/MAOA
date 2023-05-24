import os
import sys
import numpy as np
from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor


class DataCollect:
    def __init__(self):
        pass

    @staticmethod
    def get_path():
        ranked_path = 'ranked'
        each_rank_dir = [os.path.join(ranked_path, x) for x in os.listdir(ranked_path)]
        each_rank_dir = sorted(each_rank_dir, key=lambda x: int(x.split('/')[1]))
        each_out_path = [os.path.join(x, 'aims.out') for x in each_rank_dir]
        return each_rank_dir, each_out_path

    def get_contents(self, path_to_out):
        with open(path_to_out, 'r') as f:
            self.content = f.readlines()
            no_atom_marker = '| Number of atoms'
            #no_scf_marker = '| Number of relaxation steps' 
            no_scf_marker = '| Number of SCF (re)initializations          :'
            init_struc_marker = '| Atomic structure:'
            self.atoms = []
            for numi, i in enumerate(self.content):
                if no_atom_marker in i:
                    self.no_atoms = int(i.split()[-1])
                elif no_scf_marker in i:
                    self.no_scf = int(i.split()[-1])
                elif init_struc_marker in i:
                    init_struc = self.content[numi+2: numi+self.no_atoms+2] 
                    for j in init_struc:
                        atom = j.split()[3]
                        self.atoms.append(atom)
        return None 



    def get_energies(self):
        self. energies = []
        marker = '| Total energy corrected        :'
        for i in self.content:
            if marker in i:
                self.energies.append(float(i.split()[5]))
        return self.energies


    def get_forces(self):
        marker = 'Total atomic forces (unitary forces cleaned) [eV/Ang]'
        start_index = None
        self.forces = np.zeros((self.no_scf, self.no_atoms, 3))
        cnt = 0
        for numi, i in enumerate(self.content):
            if marker in i:
                start_index = numi + 1
            elif start_index is not None and i.strip() == '':
                end_index = numi 
                force = self.content[start_index:end_index]
                for numj, j in enumerate(force):
                    numbers = [float(x) for x in j.split()[-3:]]
                    self.forces[cnt, numj] = numbers
                start_index = None
                cnt += 1
        return None 

    def merge_geo_force(self):
        print(self.atoms)
        self.atom_label = np.reshape(np.array(self.atoms), (self.no_atoms, -1))
        print()
        for i in range(self.no_scf):
            print(self.no_atoms)
            print(self.energies[i])
            atom_coord_force = np.concatenate([self.atom_label, self.geo[i], self.forces[i]], axis=1)
            print(atom_coord_force)


if __name__ == "__main__":

    app_output = './aims.out'
    
    ex = extractor()
    
    ex.set_output_filepath(app_output)
    print()
    print("SCF converged geometries")
    print(ex.get_geo)
    print()
    print("SCF converged forces")
    print(ex.get_forces)
    print()
    print("Total no of SCF (re)initializations")
    print(ex.get_number_of_scf_blocks)
    print()

