'''
Author: Dong-Gi Kang
Prepare ML-IP data using FHI-aims output
Training data type: vibrational mode of a cluster
'''
import os
import sys
import numpy as np
from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor

class ML_train_generator(extractor):
    
    def __init__(self, app_version='22', tag=None):
        app_output = './aims.out'

        self.extractor = extractor()
        self.extractor.set_output_filepath(app_output)
        
        self.species = self.extractor.get_species
        self.no_atoms = self.extractor.get_no_atoms
        self.geometries = self.extractor.get_geometries
        self.order = self.extractor.get_atom_order
        self.forces = self.extractor.get_forces

        try:
            self.vib_eigvecs = self.extractor.get_vib_eigvec
        except:
            pass

        self.ucl_id = 'uccatka'
        self.job_time = '2:00:00'
        self.job_name = 'test'
        self.memory = '1'
        self.cpu_core = '40'  # for Young 40 core = 1 node
        self.payment = 'Gold'
        self.budgets = 'UCL_chemM_Woodley'
        self.path_binary = '/home/uccatka/software/fhi-aims.221103/build/aims.221103.scalapack.mpi.x' 
        self.path_fhiaims_species = '/home/uccatka/software/fhi-aims.221103/species_defaults/defaults_2020/light'
        self.step_size = 0.05
        return None


    @property
    def mod_xyz_w_vib(self):
        ''' Modify LM geometry to array of vibrational mode frames '''
        Lambda = len(np.arange(-1, 1+self.step_size, self.step_size)) * self.no_atoms*3
        self.mod_sp = np.zeros((Lambda, self.no_atoms, 3))
        cnt = 0
        for i in range(self.no_atoms * 3):
            for numj, j in enumerate(np.arange(-1, 1+self.step_size, self.step_size)):
                j = np.round(j, 2)
                frame = self.geometries[-1] + self.vib_eigvecs[i] * j
                self.mod_sp[cnt] = np.round(frame, 8)
                cnt += 1
        self.mod_sp = np.reshape(self.mod_sp, (self.no_atoms*3, len(np.arange(-1, 1+self.step_size, self.step_size)), self.no_atoms, 3))
        return self.mod_sp 


    @property
    def geometry_for_sp(self):
        ''' Convert the modified geometry (mod_xyz_w_vib) to {geometry.in} format for FHI-aims '''
        placer = np.full((self.no_atoms, 1), 'atom')
        placer_species = np.reshape(self.order, (-1, 1))
        shape = np.shape(self.mod_sp)
        self.for_sp = np.empty((shape[0], shape[1], self.no_atoms, 5), dtype=object)

        for i in range(shape[0]):
            for j in range(shape[1]):
                form = np.concatenate((placer, self.mod_sp[i][j], placer_species), axis=1)
                self.for_sp[i][j] = form
        return self.for_sp 


    @property
    def xyz_from_opti(self):
        ''' prepare training data from every SCF converged cycles of a optimisation '''
        train_xyz = 'xyz_from_opti.xyz'
        exist = [x for x in os.listdir('./') if train_xyz in x]
        if len(exist) != 0:
            os.remove(exist[0])
        else: pass
        for i in range(len(self.extractor.set_scf_blocks)):
            self.energy = self.extractor.get_total_energy(i) 
            self.geometry = self.geometries[i]
            self.force = self.forces[i]
            xyz = np.round(np.concatenate((self.geometry, self.force), axis=1), 9)
            xyz = np.concatenate((self.order, xyz), axis=1)
           
            with open('xyz_from_opti.xyz', 'a') as f:
                f.write(f'{self.no_atoms}\n')
                f.write(f'Properties-species:S:1:pos:R:3:forces:R:3 energy={self.energy} pbc="F F F"\n')
                np.savetxt(f, xyz, fmt="%s", delimiter="    ")
        print(f"total of {i+1} SCF converged structures are prepared in {train_xyz}") 


    def make_sp_control(self, path):
        ''' Write {control.in} file '''
        basis_set_files = [os.path.join(self.path_fhiaims_species, x) for x in os.listdir(self.path_fhiaims_species)]
        basis_set_all = [x.split('_')[1] for x in os.listdir(self.path_fhiaims_species)]
        basis_set_index = [basis_set_all.index(x) for x in basis_set_all if x in self.species] 
        
        path = os.path.join(path, 'control.in')
        with open(path, 'a') as f:
            f.write("#\n")
            f.write("xc                 pbesol\n")
            f.write("spin               none\n")
            f.write("relativistic       atomic_zora scalar\n")
            f.write("charge             0.\n\n")
            f.write("#  SCF convergence\n")
            f.write("occupation_type    gaussian 0.01\n")
            f.write("mixer              pulay\n")
            f.write("n_max_pulay        10\n")
            f.write("charge_mix_param   0.5\n")
            f.write("sc_accuracy_rho    1E-5\n")
            f.write("sc_accuracy_eev    1E-3\n")
            f.write("sc_accuracy_etot   1E-6\n")
            f.write("sc_accuracy_forces 1E-4\n")
            f.write("sc_iter_limit      1500\n")
            f.write("#  Relaxation\n\n")
            #f.write("relax_geometry   bfgs 1.e-3\n")
            for i in basis_set_index:
                with open(basis_set_files[i], 'r') as ff:
                    lines = ff.read()
                f.write(lines)
                f.write('\n')
        return None 


    def make_job_submit(self, path):
        ''' Write 'submit.sh' job script for SGE system '''
        _, last_part = os.path.split(path)
        _, second_last_part = os.path.split(os.path.dirname(path))
        
        # Combine the last two parts
        last_two_parts = f"{second_last_part}_{last_part}"

        path = os.path.join(path, 'submit.sh')
        with open(path, 'a') as f:
            f.write("#!/bin/bash -l\n")
            f.write('\n')
            f.write("#$ -S /bin/bash\n")
            f.write(f"#$ -l h_rt={self.job_time}\n")
            f.write(f"#$ -l mem={self.memory}G\n")
            f.write(f"#$ -N p{last_two_parts}\n")
            f.write(f"#$ -pe mpi {self.cpu_core}\n")
            f.write("#$ -cwd\n")
            f.write("\n")
            f.write(f"#$ -P {self.payment}\n")
            f.write(f"#$ -A {self.budgets}\n")

            f.write("module load gerun\n")
            f.write("module load userscripts\n")
            f.write("module unload -f compilers mpi gcc-libs\n")
            f.write("module load gcc-libs/4.9.2\n")
            f.write("module unload -f compilers mpi\n")
            f.write("module load beta-modules\n")
            f.write("module load openblas/0.3.7-serial/gnu-4.9.2\n")
            f.write("module load compilers/intel/2019/update5\n")
            f.write("module load mpi/intel/2018/update3/intel\n")

            f.write("\n")
            f.write("#$ -m e\n")
            f.write(f"#$ -M {self.ucl_id}@ucl.ac.uk\n")
            f.write("\n")
            f.write(f"gerun {self.path_binary} > aims.out\n")





import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigenvector", type=str, help="A string of space-separated eigenvector indices. For example, '7 8 9 10'")
    args = parser.parse_args()

    # Split the string of indices into a list of integers
    indices = list(map(int, args.eigenvector.split()))

    app_output = './aims.out'
    step_size = 0.05
    ml = ML_train_generator()

    ml.mod_xyz_w_vib
    sp_frame = ml.geometry_for_sp
    shape = np.shape(sp_frame)
    if not os.path.exists('sp'):
        os.mkdir('sp') 
    else: pass
    
    for i in indices:  # Now we only iterate over the specified indices
        if not os.path.exists(os.path.join('sp', str(i+1))):
            os.mkdir(f'sp/{str(i)}')
        else: pass

        for numj, j in enumerate(np.arange(-1, 1+step_size, step_size)):
            j = str(np.round(j, 2))
            os.mkdir(f'sp/{str(i)}/lambda_{j}')
            with open(f'sp/{i}/lambda_{j}/geometry.in', 'w') as f:
                for row in sp_frame[i-1][numj]:
                    line = ' '.join(str(x) for x in row)
                    f.write(line + '\n')
            ml.make_sp_control(f'sp/{i}/lambda_{j}')
            ml.make_job_submit(f'sp/{i}/lambda_{j}')
            os.chdir(f'sp/{i}/lambda_{j}')
            os.system('qsub submit.sh')
            os.chdir('../../../')


