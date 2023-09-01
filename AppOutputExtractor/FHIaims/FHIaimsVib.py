from AppOutputExtractor.OutputExtractor import BaseExtractor
from AppOutputExtractor.FHIaims.FHIaimsMolecule import molecule as fmol
from AppOutputExtractor.FHIaims.FHIaimsMolecule import calculate_rmsd_molecules
from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor

from ShellCommand import shellcommand
import ParsingSupport

import os
import shutil
import string,json


class aimsvibcalc(BaseExtractor):

    def __init__(self, app_version='22', tag=None):
        '''
        '''
        app_output = './aims.out'
        self.extractor = extractor()
        self.extractor.set_output_filepath(app_output)
        self.no_atoms = self.extractor.get_no_atoms()
        self.geometries = self.extractor.get_geometries(self.no_atoms)
        self.order = self.extractor.get_atom_order(self.no_atoms)
        self.species = self.extractor.get_species(self.order)


        self.ucl_id = 'uccatka'
        self.job_time = '2:00:00'
        #self.job_name = 'test'
        self.memory = '2'
        self.cpu_core = '40'  # for Young 40 core = 1 node
        self.payment = 'Gold'
        self.budgets = 'UCL_chemM_Woodley'
        self.path_binary = '/home/uccatka/software/fhi-aims.221103/build/aims.221103.scalapack.mpi.x'
        self.vib_path_binary = '/home/uccatka/software/fhi-aims.221103/build/src/vibrations/numerical_vibrations.pl'
        self.path_fhiaims_species = '/home/uccatka/software/fhi-aims.221103/species_defaults/defaults_2020/light'
        self.step_size = 0.05

        super().__init__(app='FHIaims',version=app_version)

        # set app output patterns
        module_path = os.path.dirname(os.path.abspath(__file__)) + '/OutputPattern' # getting this module path, '__file__'
        self.patterns = self.load_patterns(module_path)

        # memo
        self.tag = tag

        # shellcommand obj
        self.shell = shellcommand()

        return None

    def make_job_submit(self, job_name, loc='./vibration', step_size='0.0025'):
        ''' Write 'submit.sh' job script for SGE system '''
        path = os.path.join(loc, 'submit.sh')
        with open(path, 'a') as f:
            f.write("#!/bin/bash -l\n")
            f.write("\n")
            f.write("#$ -S /bin/bash\n")
            f.write(f"#$ -l h_rt={self.job_time}\n")
            f.write(f"#$ -l mem={self.memory}G\n")
            f.write(f"#$ -N {job_name}\n")
            f.write(f"#$ -pe mpi {self.cpu_core}\n")
            f.write("#$ -cwd\n")
            f.write("\n")
            f.write(f"#$ -P {self.payment}\n")
            f.write(f"#$ -A {self.budgets}\n")
            f.write("\n")
            f.write("#$ -m e\n")
            f.write(f"#$ -M {self.ucl_id}@ucl.ac.uk\n")
            f.write("\n")
            f.write("module purge\n")
            f.write("module load gerun\n")
            f.write("module load userscripts\n")
            f.write("module load gcc-libs/4.9.2\n")
            f.write("module unload -f compilers mpi\n")
            f.write("module load beta-modules\n")
            f.write("module load gcc-libs/10.2.0\n")
            f.write("module load openblas/0.3.7-serial/gnu-4.9.2\n")
            f.write("module load compilers/intel/2019/update5\n")
            f.write("module load mpi/intel/2018/update3/intel\n")
            f.write("module load cmake/3.21.1\n\n")            
            
            f.write(f"{self.vib_path_binary} {job_name}_{step_size} {step_size} > vibres.out\n")

    @property
    def vib_calc_prep(self):
        #shutil.copy('./control.in', './vibration')        
        geo_next = 'geometry.in.next_step'
        geo = 'geometry.in'
        vib_dir = 'vibration'
        geometry_files = [x for x in os.listdir('./') if '.in' in x]
        if os.path.exists(geo_next):
            shutil.copy(geo_next, f'{vib_dir}/{geo}')
            shutil.copy('hessian.aims', vib_dir)
        else:
            shutil.copy(geo, vib_dir)

        basis_set_files = [os.path.join(self.path_fhiaims_species, x) for x in os.listdir(self.path_fhiaims_species)]
        basis_set_all = [x.split('_')[1] for x in os.listdir(self.path_fhiaims_species)]
        basis_set_index = [basis_set_all.index(x) for x in basis_set_all if x in self.species]

        with open(os.path.join(vib_dir, 'control.in'), 'a') as f:
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
            f.write("sc_iter_limit      1500\n\n")

            for i in basis_set_index:
                with open(basis_set_files[i], 'r') as ff:
                    lines = ff.read()
                f.write(lines)
                f.write('\n')
        return None

if __name__ == "__main__":
    vib = aimsvibcalc()
    os.mkdir('vibration')
    vib.vib_calc_prep
    current_dir_name = os.path.basename(os.getcwd())
    vib.make_job_submit(f'n{current_dir_name}')
    os.chdir('vibration')
    os.system('qsub submit.sh')



