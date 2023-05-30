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
        self.memory = '2'
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
        Lambda = len(np.arange(-1, 1, self.step_size))
        self.mod_sp = np.zeros((Lambda, self.no_atoms, 3))
        for i in range(self.no_atoms*3):
            for numj, j in enumerate(np.arange(-1, 1, self.step_size)):
                frame = self.geometries[-1] + self.vib_eigvecs[i] * j
                self.mod_sp[numj] = frame
        return self.mod_sp 


    @property
    def geometry_for_sp(self):
        ''' Convert the modified geometry (mod_xyz_w_vib) to {geometry.in} format for FHI-aims '''
        placer = np.full((self.no_atoms, 1), 'atom')
        placer_species = np.reshape(self.order, (-1, 1))
        self.for_sp = np.empty((np.shape(self.mod_sp)[0], self.no_atoms, 5), dtype=object)
        for numi, i in enumerate(self.mod_sp):
            form = np.concatenate((placer, i, placer_species), axis=1)
            self.for_sp[numi] = form
        return self.for_sp 


    @property
    def xyz_from_opti(self):
        ''' prepare training data from every SCF converged cycles of a optimisation '''
        train_xyz = 'xyz_from_opti.xyz'
        exist = [x for x in os.listdir('./') if train_xyz in x]
        if len(exist) != 0:
            os.remove(exist[0])
        else: pass

        for numi, i in enumerate(range(len(self.extractor.set_scf_blocks))):
            #for numj, j in enumerate(i):
            self.energy = self.extractor.get_total_energy(i) 
            self.geometry = self.geometries(i)
            self.force = self.forces(i)
            xyz = np.round(np.concatenate((self.geometry, self.force), axis=1), 9)
            xyz = np.concatenate((self.order, xyz), axis=1)
           
            with open('xyz_from_opti.xyz', 'a') as f:
                f.write(f'{self.no_atoms}' + '\n')
                f.write(f'{self.energy}' + '\n')
                np.savetxt(f, xyz, fmt="%s")
        print(f"total of {i+1} SCF converged structures are prepared in {train_xyz}") 


    def make_sp_control(self):
        ''' Write {control.in} file '''
        basis_set_files = [os.path.join(self.path_fhiaims_species, x) for x in os.listdir(self.path_fhiaims_species)]
        basis_set_all = [x.split('_')[1] for x in os.listdir(self.path_fhiaims_species)]
        basis_set_index = [basis_set_all.index(x) for x in basis_set_all if x in self.species] 
        
        with open('control.in', 'a') as f:
            f.write("xc                 pbesol\n")
            f.write("spin               none\n")
            f.write("relativistic       atomic_zora scalar\n")
            f.write("charge             0.\n")
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
            f.write("#  Relaxation\n")
            f.write("\n")
            #f.write("relax_geometry   bfgs 1.e-3\n")
            for i in basis_set_index:
                with open(basis_set_files[i], 'r') as ff:
                    lines = ff.read()
                f.write(lines)
                f.write('\n')
        return None 


    def make_job_submit(self):
        ''' Write 'submit.sh' job script for SGE system '''
        with open('submit.sh', 'a') as f:
            f.write("#!/bin/bash -l\n")
            f.write("# Batch script to run an MPI parallel job with the upgraded software\n")
            f.write("# stack under SGE with Intel MPI.\n")
            f.write("# 1. Force bash as the executing shell.\n")
            f.write("#$ -S /bin/bash\n")
            f.write("# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).\n")
            f.write(f"#$ -l h_rt={self.job_time}\n")
            f.write("# 3. Request 1 gigabyte of RAM per process.\n")
            f.write(f"#$ -l mem={self.memory}G\n")
            f.write("# 4. Request 15 gigabyte of TMPDIR space per node (default is 10 GB)\n")
            f.write("# 5. Set the name of the job.\n")
            f.write(f"#$ -N {self.job_name}\n")
            f.write("# 6. Select the MPI parallel environment and 24 processes.\n")
            f.write(f"#$ -pe mpi {self.cpu_core}\n")
            f.write("# 7. Set the working directory to somewhere in your scratch space.  This is\n")
            f.write("# a necessary step with the upgraded software stack as compute nodes cannot\n")
            f.write("# write to $HOME.\n")
            f.write("#$ -cwd\n")
            f.write("\n")
            f.write("# 8. Set the budget\n")
            f.write(f"#$ -P {self.payment}\n")
            f.write(f"#$ -A {self.budgets}\n")
            f.write("\n")
            f.write("# 9. Send email when the job finishes or aborts.\n")
            f.write("#$ -m e\n")
            f.write(f"#$ -M {self.ucl_id}@ucl.ac.uk\n")
            f.write("\n")
            f.write("# 10. Run our MPI job.  GERun is a wrapper that launches MPI jobs on our clusters.\n")
            f.write(f"gerun {self.path_binary} > aims.out\n")




if __name__ == "__main__":

    app_output = './aims.out'
    ml = ML_train_generator()
    #ml.mod_xyz_w_vib
    #ml.geometry_for_sp
    ml.make_sp_control()
    ml.make_job_submit()
    ml.xyz_from_opti


