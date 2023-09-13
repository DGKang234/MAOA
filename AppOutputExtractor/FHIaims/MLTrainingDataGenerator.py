
"""
dev note:
work on breathing method"""

'''
Author: Dong-Gi Kang
Prepare ML-IP data using FHI-aims output
Training data type: vibrational mode of a cluster

[it retreive the vibrational mode cluster geometry and forces from single point calculation and
generates extended xyz format of Training_set.xyz: contains the total energy, atomic coordination, atomic forces]
The Training_set.xyz will be placed in FIT directory and each vibrational mode ext xyz files are generated in ext_xyz directory (The directories are automatically generated from the code)

N.B. Change the UCL id, budget code, executable path for FHI-aims and fhi-aims species directory



help: 
python MLTrainingDataGenerator.py -h
(execute the file at the directory where the geometry.in, control.in, viration (dir) located)


1. python {code.py} --mode run --eigenvector="7 8 9 10" would grab 7th, 8th, 9th 10th (can selectively) then, modify the GM with the step_size (GM geometry + eigenvector * step_size) and prepare the individual directories and submit the single point calculations. 

2. python {code.py} --mode retrieve --eigenvector="7 8 9 10" would grab the generated data from the [1.] and make the ext xyz for each vibrational mode and store into the ext_xyz directory

3. [python {code.py} --mode make_extxyz] would grab the all data from ext_xyz and make ext xyz format of Training_set.xyz in FIT directory 

4. if you want to trianing MACE or GAP ML-IP use MACE_lib.py or second_GAP.py
'''

import os
import sys
import random
import numpy as np
import argparse
from itertools import groupby
from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor

class ML_train_generator(extractor):
    
    def __init__(self, app_version='22', tag=None):

        self.breathing_called = False

        #self.extractor = extractor()
        #self.extractor.set_output_filepath(app_output)
        #self.no_atoms = self.extractor.get_no_atoms()
        #self.geometries = self.extractor.get_geometries(self.no_atoms)
        #self.order = self.extractor.get_atom_order(self.no_atoms)
        #ID = self.extractor.get_species(self.order)
        #self.forces = self.extractor.get_forces(self.no_atoms)
        #self.vib_eigvecs = self.extractor.get_vib_eigvec(self.no_atoms)

        self.ucl_id = 'uccatka'
        self.job_time = '2:00:00'
        self.job_name = 'test'
        self.memory = '1'
        self.cpu_core = '40'  # for Young 40 core = 1 node
        self.payment = 'Gold'
        self.budgets = 'UCL_chemM_Woodley'
        self.path_binary = '/home/uccatka/software/fhi-aims.221103/build/aims.221103.scalapack.mpi.x' 
        self.path_fhiaims_species = '/home/uccatka/software/fhi-aims.221103/species_defaults/defaults_2020/light'
        self.step_size = 0.1        ##### STEP SIZE #####
        return None


    def initiate(self):
        app_output = './aims.out'
        self.extractor = extractor()
        self.extractor.set_output_filepath(app_output)
        self.no_atoms = self.extractor.get_no_atoms()
        self.geometries = self.extractor.get_geometries(self.no_atoms)
        self.order = self.extractor.get_atom_order(self.no_atoms)
        self.ID = self.extractor.get_species(self.order)
        self.forces = self.extractor.get_forces(self.no_atoms)
        self.vib_eigvecs = self.extractor.get_vib_eigvec(self.no_atoms)


    def mod_xyz_w_vib(self):
        ''' Modify LM geometries to the frames of vibrational mode frames '''
        Lambda = len(np.arange(-1, 1+self.step_size, self.step_size)) * self.no_atoms*3
        self.mod_sp = np.zeros((Lambda, self.no_atoms, 3))
        cnt = 0
        for i in range(self.no_atoms * 3): # 3N dimension
            for numj, j in enumerate(np.arange(-1, 1+self.step_size, self.step_size)):  # -1 to 1 in every step size
                j = np.round(j, 2)
                frame = self.geometries[-1] + self.vib_eigvecs[i] * j
                self.mod_sp[cnt] = np.round(frame, 8)
                cnt += 1
        self.mod_sp = np.reshape(self.mod_sp, (self.no_atoms*3, len(np.arange(-1, 1+self.step_size, self.step_size)), self.no_atoms, 3))
        return self.mod_sp 


    def mod_xyz_w_rand_pair_vib(self):

        list_eigvecs = list(range(6, self.no_atoms*3))
        random.shuffle(list_eigvecs)
        self.pairs_eigvecs = [[list_eigvecs[i], list_eigvecs[i+1]] for i in range(0, len(list_eigvecs), 2)]
        fname_pairs = [f"{x}-{y}" for x, y in self.pairs_eigvecs]

        #Lambda = len(np.arange(-1, 1+self.step_size, self.step_size)) * (self.no_atoms*3-6)  # range of steps for all vib. mode, except E(3) 
        #self.mod_sp_pair = np.zeros((Lambda, self.no_atoms, 3))

        self.mod_sp_pair = np.zeros((len(self.pairs_eigvecs), len(np.arange(-1, 1+self.step_size, self.step_size)), self.no_atoms, 3))    # range of steps for all vib. mode, except E(3) 

        cnt = 0
        for numi, i in enumerate(self.pairs_eigvecs):
            for numj, j in enumerate(np.arange(-1, 1+self.step_size, self.step_size)):
                j = np.round(j, 2)
                frame = self.geometries[-1] + (self.vib_eigvecs[i[0]]+self.vib_eigvecs[i[1]]) * j
                #self.mod_sp_pair[cnt] = np.round(frame, 8)
                self.mod_sp_pair[numi][numj] = np.round(frame, 8)
                cnt += 1 
        
        self.mod_sp_pair = np.reshape(self.mod_sp_pair, (len(self.pairs_eigvecs), numj+1, self.no_atoms, 3))
        return self.mod_sp_pair, fname_pairs 


    def breathing(self):
        scale = np.arange(0.6, 1+self.step_size, self.step_size)
        Lambda = len(scale) #* self.no_atoms*3
        self.mod_sp_breath = np.zeros((Lambda, self.no_atoms, 3))
        # shift the centre of mass of the structure to (0, 0, 0)
        coord = self.geometries[0]
        com = coord.sum(axis=0)
        com = com / int(self.no_atoms)
        coord_x = np.subtract(coord[:, 0], com[0], out=coord[:, 0])
        coord_y = np.subtract(coord[:, 1], com[1], out=coord[:, 1])
        coord_z = np.subtract(coord[:, 2], com[2], out=coord[:, 2])
        coord = list(zip(coord_x, coord_y, coord_z))
        coord = np.array(coord)
        cnt = 0
 
        for numj, j in enumerate(scale):
            j = np.round(j, 2)
            frame = coord * j
            self.mod_sp_breath[cnt] = np.round(frame, 8)
            cnt += 1
        
        self.mod_sp_breath = np.reshape(self.mod_sp_breath, (len(scale), self.no_atoms, 3)) 
        self.breathing_called = True
        return self.mod_sp_breath, scale


    #@property
    def geometry_for_sp(self, mod_sp):
        ''' Convert the modified geometry (mod_xyz_w_vib) to {geometry.in} format for FHI-aims '''
        # vibrational modes
        if not self.breathing_called:
            print("@@@@@@@")
            placer = np.full((self.no_atoms, 1), 'atom')
            placer_species = np.reshape(self.order, (-1, 1))
            shape = np.shape(mod_sp)
            self.for_sp = np.empty((shape[0], shape[1], self.no_atoms, 5), dtype=object)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    form = np.concatenate((placer, mod_sp[i][j], placer_species), axis=1)
                    self.for_sp[i][j] = form
            return self.for_sp, self.no_atoms

        # breathing mode
        else:
            print("*******")
            placer_breath = np.full((self.no_atoms, 1), 'atom')
            placer_species_breath = np.reshape(self.order, (-1, 1))
            shape_breath = np.shape(mod_sp)
            self.for_sp = np.empty((shape_breath[0], shape_breath[1], 5), dtype=object)
            for i in range(shape_breath[0]):
                form = np.concatenate((placer_breath, mod_sp[i], placer_species_breath), axis=1)
                self.for_sp[i] = form 
            return self.for_sp, self.no_atoms


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
                f.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={self.energy} pbc="F F F"\n')
                np.savetxt(f, xyz, fmt="%s", delimiter="    ")
        print(f"total of {i+1} SCF converged structures are prepared in {train_xyz}") 


    def make_sp_control(self, path):
        ''' Write {control.in} file '''
        basis_set_files = [os.path.join(self.path_fhiaims_species, x) for x in os.listdir(self.path_fhiaims_species)]
        basis_set_all = [x.split('_')[1] for x in os.listdir(self.path_fhiaims_species)]
        basis_set_index = [basis_set_all.index(x) for x in basis_set_all if x in self.ID] 
        
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
            f.write("####$ -m e\n")
            f.write(f"####$ -M {self.ucl_id}@ucl.ac.uk\n")
            f.write("\n")
            f.write(f"gerun {self.path_binary} > aims.out\n")


    @staticmethod
    def sorting_key(path):
        parts = path.split('/')
        second_key = int(parts[1]) if parts[1] != "breathing" else float('inf')
        third_key = float(parts[2].split('_')[1])  # Consider lambda value regardless of the second part
        return second_key, third_key

    def retrieve_results_c(self, eigenvectors):
        print("---retrieve---")
        eigvec_path = [os.path.join('sp', str(eigvec)) for eigvec in eigenvectors]
        sp_path = [os.path.join(dirpath, fname) for dirpath in eigvec_path for fname in os.listdir(dirpath)]
        lambda_path = [os.path.join(dirpath, fname) for dirpath in sp_path for fname in os.listdir(dirpath) if fname == 'aims.out']
        aims_out_path = sorted(lambda_path, key=self.sorting_key)
        aims_out_path = [list(group) for key, group in groupby(aims_out_path, lambda x: x.split('/')[1])]

        if not os.path.exists('ext_xyz'):
            os.mkdir('ext_xyz')
        cnt = 0 
        for numi, i in enumerate(aims_out_path):
                filename = f"ext_xyz/ext_{i[0].split('/')[1]}_eigv.xyz"       #
                with open(filename, 'a') as f:                  #
                    for j in i:
                        ex = extractor() 
                        ex.set_output_filepath(j)
                        #ex.set_scf_blocks
    
                        no_atoms = ex.get_no_atoms()
                        geometries, atom_label = ex.get_sp_geometries(j)
                        forces = ex.get_sp_forces(no_atoms, j)
                        total_energy = ex.get_sp_total_energy(j)

                        coulomb_E, coulomb_F = self.coulomb_E_F(atom_label, geometries)

                        # subtract coulomb energy and force
                        energy = total_energy - coulomb_E
                        forces = forces - coulomb_F                        
                        form = np.concatenate((ex.get_sp_atom_order(), geometries, forces), axis=1)

                        f.write(str(no_atoms) + '\n')
                        f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc="F F F"\n')
                        np.savetxt(f, form, fmt="%s", delimiter="        ")


    def retrieve_results(self, eigenvectors):
        print("---retrieve---")
        eigvec_path = [os.path.join('sp', str(eigvec)) for eigvec in eigenvectors]
        sp_path = [os.path.join(dirpath, fname) for dirpath in eigvec_path for fname in os.listdir(dirpath)]
        lambda_path = [os.path.join(dirpath, fname) for dirpath in sp_path for fname in os.listdir(dirpath) if fname == 'aims.out']
        aims_out_path = sorted(lambda_path, key=self.sorting_key)
        aims_out_path = [list(group) for key, group in groupby(aims_out_path, lambda x: x.split('/')[1])]

        if not os.path.exists('ext_xyz'):
            os.mkdir('ext_xyz')
        cnt = 0
        for numi, i in enumerate(aims_out_path):
                filename = f"ext_xyz/ext_{i[0].split('/')[1]}_eigv.xyz"       #
                with open(filename, 'a') as f:                  #
                    for j in i:
                        ex = extractor()
                        ex.set_output_filepath(j)
                        #ex.set_scf_blocks


                        no_atoms = ex.get_no_atoms()
                        geometries, atom_label = ex.get_sp_geometries(j)
                        forces = ex.get_sp_forces(no_atoms, j)
                        total_energy = ex.get_sp_total_energy(j)

                        ## subtract coulomb energy and force
                        print("Coulomb interactions are eliminated")
                        coulomb_E, coulomb_F = self.coulomb_E_F(atom_label, geometries)
                        total_energy = total_energy - coulomb_E
                        forces = forces - coulomb_F
                        form = np.concatenate((ex.get_sp_atom_order(), geometries, forces), axis=1)

                        f.write(str(no_atoms) + '\n')
                        f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={total_energy} pbc="F F F"\n')
                        np.savetxt(f, form, fmt="%s", delimiter="        ")



    def make_extxyz(self):
        if not os.path.exists('FIT'):
            os.mkdir('FIT')
        else: pass
        if os.path.exists('./FIT/Training_set.xyz'):
            os.remove('./FIT/Training_set.xyz')
            print("You may want to check the .xyz files in the FIT")
        else:
            with open('FIT/Training_set.xyz', 'a') as outfile:
                filenames = [file for file in os.listdir('ext_xyz') if file.endswith('.xyz')]
                sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[1]) if x.split('_')[1] != 'breathing' else float('inf'))
                for file in sorted_filenames:
                    print(file)
                    with open(os.path.join('ext_xyz', file), 'r') as infile:
                        for line in infile:
                            outfile.write(line)


    def split_xyz_file(self, input_file, train_file, valid_file, test_file):
        with open(input_file, 'r') as infile:
            train_out = open(train_file, 'w')
            valid_out = open(valid_file, 'w')
            test_out = open(test_file, 'w')
    
            block_counter = 0
            line = infile.readline()
    
            while line:
                if line.strip().isdigit():  
                    no_atoms = int(line.strip())
                    block_lines = [line] + [infile.readline() for _ in range(no_atoms + 1)]  # Read the block
                    
                    if block_counter % 5 < 3:
                        output_file = train_out
                    elif block_counter % 5 == 3:
                        output_file = valid_out
                    else:
                        output_file = test_out
    
                    output_file.writelines(block_lines)
                    block_counter += 1
                
                line = infile.readline()
    
            train_out.close()
            valid_out.close()
            test_out.close()


    def coulomb_energy(self, r, cat_q, an_q):
        return (cat_q * an_q) / r * 14.3996439067522

    def coulomb_force(self, r, unit_r, cat_q, an_q):
        return (cat_q * an_q) / r**2 * unit_r * 14.3996439067522

    def coulomb_E_F(self, atom_label, structure):
        self.charges = {"Al": 3, "F": -1}
        coulomb_e = 0.0
        forces = np.zeros_like(structure)

        for i in range(len(structure)):
            for j in range(i+1, len(structure)):
                coord1 = structure[i]
                coord2 = structure[j]
                atom1 = atom_label[i][0]
                atom2 = atom_label[j][0]

                r_vec = coord2 - coord1
                r = np.linalg.norm(r_vec)
                unit_r = r_vec / r              # unit vec

                # energy 
                energy_pair = self.coulomb_energy(r, self.charges[atom1], self.charges[atom2])
                coulomb_e += energy_pair

                # force 
                force_pair = self.coulomb_force(r, unit_r, self.charges[atom1], self.charges[atom2])

                # add forces to atoms
                forces[i] -= force_pair
                forces[j] += force_pair

        return coulomb_e, forces






# executing the code using the class
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigenvector", type=str, help="A string of space-separated eigenvector indicies. For example, '7 8 9 10'")
    parser.add_argument("--mode", type=str, choices=["run", "run_pair", "breath", "retrieve", "retrieve_c", "make_extxyz", "make_extxyz_"], default="run", help="Specify 'run' to execute the first part of the code, 'retrieve' to execute the second part of the code, or 'make_extxyz' to append all .xyz files into Training_set.xyz.")
    args = parser.parse_args()

    ml = ML_train_generator()
    step_size = ml.step_size    ##### STEP SIZE #####


    #
    # run
    #
    if args.mode == "run":
        ml.initiate()
        app_output = './aims.out'
       
        mod_sp = ml.mod_xyz_w_vib()                        # for each of vib. mode
        sp_frame, no_atoms = ml.geometry_for_sp(mod_sp)

        shape = np.shape(sp_frame)
        if args.eigenvector == 'all':
            indicies = list(range(7, no_atoms*3+1))
            print("all eigenvectors without rotational and translational\n")
        else:
            indicies = list(map(int, args.eigenvector.split()))

        if not os.path.exists('sp'):
            os.mkdir('sp')
        else: pass
   
        for i in indicies:  # Now we only iterate over the specified indicies
            if not os.path.exists(os.path.join('sp', str(i))):
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
                os.system('qsub submit.sh')                     # submit jobs
                os.chdir('../../../')

    #
    # randomly pair up eigenvectors
    #
    elif args.mode == "run_pair":
        ml.initiate()
        app_output = './aims.out'
        mod_sp, fname_pairs = ml.mod_xyz_w_rand_pair_vib()                 # randomly paired vib. mode
        sp_frame, no_atoms = ml.geometry_for_sp(mod_sp)
        shape = np.shape(sp_frame)

        if args.eigenvector == 'all':
            #indicies = list(range(15))
            indicies = list(range(3*no_atoms))[6:]
            print("all paired eigenvectors without E(3), (rotational and translational)\n")
        else:
            indicies = list(map(int, args.eigenvector.split()))
        
        if not os.path.exists('sp'):
            os.mkdir('sp')
        else: pass

        for i in range(int(len(indicies)/2)):  # Now we only iterate over the specified indicies
            #i = i+1 
            if not os.path.exists(os.path.join('sp', fname_pairs[i])):
                os.mkdir(f'sp/{fname_pairs[i]}_pair')
            else: pass

            for numj, j in enumerate(np.arange(-1, 1+step_size, step_size)):
                j = str(np.round(j, 2))
                os.mkdir(f'sp/{fname_pairs[i]}_pair/lambda_{j}')
                with open(f'sp/{fname_pairs[i]}_pair/lambda_{j}/geometry.in', 'w') as f:
                    for row in sp_frame[i][numj]:
                        line = ' '.join(str(x) for x in row)
                        f.write(line + '\n')
                ml.make_sp_control(f'sp/{fname_pairs[i]}_pair/lambda_{j}')
                ml.make_job_submit(f'sp/{fname_pairs[i]}_pair/lambda_{j}')
                os.chdir(f'sp/{fname_pairs[i]}_pair/lambda_{j}')
                os.system('qsub submit.sh')                     # submit jobs
                os.chdir('../../../')

    #
    # preparen and run breathing mode single point calc
    #
    if args.mode == "breath":
        ml.initiate()
        if not os.path.exists('sp'):
            os.mkdir('sp')
        if not os.path.exists('sp/breathing'):
            os.mkdir('sp/breathing')                           # for breathing mode
      
        mod_sp_breath, scale = ml.breathing()
        #print(mod_sp_breath)
        sp_frame, no_atoms = ml.geometry_for_sp(mod_sp_breath)

        # breathing
        for numk, k in enumerate(scale):
            k = str(np.round(k, 2))
            os.mkdir(f'sp/breathing/lambda_{k}')

            with open(f'sp/breathing/lambda_{k}/geometry.in', 'w') as f:
                for row in sp_frame[numk]: 
                    line = ' '.join(str(x) for x in row)
                    f.write(line + '\n')
            ml.make_sp_control(f'sp/breathing/lambda_{k}')
            ml.make_job_submit(f'sp/breathing/lambda_{k}')
            os.chdir(f'sp/breathing/lambda_{k}')
            os.system('qsub submit.sh')                     # submit job
            os.chdir('../../../')

    #
    # Collect data from single point calculated data
    #
    elif args.mode == "retrieve":
        ml.initiate()
        no_atoms = ml.no_atoms
        if args.eigenvector == 'all':
            indicies = list(range(7, no_atoms*3+1))
            indicies.append('breathing')
            print(indicies)
            print("all eigenvectors without rotational and translational\n")
        else:
            indicies = list(args.eigenvector.split())
            indicies = [int(x) if x.isdigit() else x for x in indicies]
            print(indicies)
        ml.retrieve_results(indicies)
   
    # collect coulomb subtracted data 
    elif args.mode == "retrieve_c":
        ml.initiate()
        no_atoms = ml.no_atoms
        if args.eigenvector == 'all':
            indicies = list(range(7, no_atoms*3+1))
            indicies.append('breathing')
            print(indicies)
            print("all eigenvectors without rotational and translational\n")
        else:
            indicies = list(args.eigenvector.split())
            indicies = [int(x) if x.isdigit() else x for x in indicies]
            print(indicies)
        ml.retrieve_results_c(indicies)

    # 
    # make training data and split to train, test, valid data 
    #
    elif args.mode == "make_extxyz":
        ml.make_extxyz()
        print("splitting training, test, validation data")
        ml.split_xyz_file('./FIT/Training_set.xyz', './FIT/Training_set_test.xyz', './FIT/Validation_set_test.xyz', './FIT/Testing_set_test.xyz')

    # dev
    elif args.mode == "make_extxyz_":
        ml.split_xyz_file('./Training_set.xyz', './Training_set_test.xyz', './Validation_set_test.xyz', './Testing_set_test.xyz')


