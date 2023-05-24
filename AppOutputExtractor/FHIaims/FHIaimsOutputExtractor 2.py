#

from AppOutputExtractor.OutputExtractor import BaseExtractor
from AppOutputExtractor.FHIaims.FHIaimsMolecule import molecule as fmol
from AppOutputExtractor.FHIaims.FHIaimsMolecule import calculate_rmsd_molecules

from ShellCommand import shellcommand
import ParsingSupport

import os,re
import numpy as np
import string,json

class extractor(BaseExtractor):

    def __init__(self,app_version='22',tag=None):
        '''
        '''
        super().__init__(app='FHIaims',version=app_version)

        # set app output patterns
        module_path = os.path.dirname(os.path.abspath(__file__)) + '/OutputPattern' # getting this module path, '__file__'
        self.patterns = self.load_patterns(module_path)

        # memo
        self.tag = tag

        # shellcommand obj
        self.shell = shellcommand()

    def set_output_filepath(self,path):

        if os.path.exists(path):
            self.output_filepath = path
        else:
            self.output_filepath = None
            print('in {} method "set_output_filepath()", cannot find the file at: "{}" '.format(__file__,path))

    def set_input_geometry_filepath(self,path):
    
        if os.path.exists(path):
            self.input_geometry_filepath = path
            self.input_geometry = fmol(path)
        else:
            self.input_geometry_filepath = None
            print('in {} method "set_input_geometry_filepath()", cannot find the file at: "{}" '.format(__file__,path))

    def set_output_geometry_filepath(self,path):

        if os.path.exists(path):
            self.output_geometry_filepath = path
            self.output_geometry = fmol(path)
        else:
            self.output_geometry_filepath = None
            print('in {} method "set_output_geometry_filepath()", cannot find the file at: "{}" '.format(__file__,path))
    
    def check_filepaths(self):
        #print('App output     : {}'.format(self.output_filepath))
        #print('geometry input : {}'.format(self.input_geometry_filepath))
        #print('geometry output: {}'.format(self.output_geometry_filepath)) 
        '''
            field: (0) AppOutput (1) InputGeometry (2) OutputGeometry
        '''
        return [self.output_filepath,self.input_geometry_filepath,self.output_geometry_filepath]

    def get_input_molecule(self):
        checker = self.check_filepaths()[1]
        if checker:
            return self.input_geometry
        else:
            print('input geometry is not loaded!')

    def get_output_molecule(self):
        checker = self.check_filepaths()[2]
        if checker:
            return self.output_geometry
        else:
            print('output geometry is not loaded!')

    '''
        Interaction with app output file
    '''

    def check_calculation_success(self):

        self.shell.set_tarfile(self.output_filepath)
        cmd = self.shell.grep(self.patterns['SUCCESS']['pattern'])
        shell_res = self.shell.execute(cmd)

        if shell_res != None:
            self.output_success_tag = True
        else:
            self.output_success_tag = False

        return self.output_success_tag          #!!!

    def check_calculation_runtime(self):
        # wall clock time
        self.shell.set_tarfile(self.output_filepath)
        cmd = self.shell.pipe(self.shell.grep(self.patterns['APP_RUNTIME']['pattern']),self.shell.awk(self.patterns['APP_RUNTIME']['wtime_token']))
        target = self.shell.execute(cmd)    #!!!

        try:
            target = float(target)
            return target
        except:
            print('failed to get calculation wtime')
            return None

    def check_parallel_task(self):
        # used cpus
        self.shell.set_tarfile(self.output_filepath)
        cmd = self.shell.pipe(self.shell.grep(self.patterns['APP_RESOURCE_USED']['pattern']),self.shell.awk(self.patterns['APP_RESOURCE_USED']['token']))
        target = self.shell.execute(cmd)    #!!!

        try:
            target = int(target)
            return target
        except:
            print('failed to get parallel task number, recheck the app output file')
            return None
    '''
        Loading SCF converged blocks ... possibly useful for further app output collation   
    '''

    @property
    def set_scf_blocks(self) -> list:
        '''
            * special blocks:
            self.scf_converged_blocks[0] -> first SCF converged blocks [line_start,line_end]
            self.scf_converged_blocks[-1]-> final SCF converged blocks
        '''
        pattern = self.patterns['BEGIN_SCF']['pattern'].replace("'","")
        self.total_lnumber, self.scf_block_lines = ParsingSupport.find_pattern_with_last_word(self.output_filepath,pattern) # GET LINE NUMBERS OF SCF (CONVERGED) BLOCKS
    
        self.scf_converged_blocklines = []          #!!!
        self.scf_converged_blocks = []              #!!!

        # IF ITEM IN ITERABLE SAVE THE LINE NUMBEERS [START,END]
        for i, item in enumerate(self.scf_block_lines[:-1]):
            curr_tag = int(self.scf_block_lines[i][1])
            next_tag = int(self.scf_block_lines[i+1][1])

            if next_tag < curr_tag:

                block_start = self.scf_block_lines[i][0]
                block_end   = self.scf_block_lines[i+1][0]

                self.scf_converged_blocklines.append([block_start,block_end])

        # FIANL SCF CONVERGED BLOCK (BEFORE APP FINALISATION)
        block_start = self.scf_block_lines[-1][0]
        block_end   = self.total_lnumber
        self.scf_converged_blocklines.append([block_start,block_end])

        # SAVE THE BLOCKS ... 'self.scf_converged_blocks' -> python list
        for item in self.scf_converged_blocklines:
            self.scf_converged_blocks.append(ParsingSupport.get_lines(self.output_filepath,item[0],item[1]))
        #for numi, i in enumerate(self.scf_converged_blocks):
        #    for numj, j in enumerate(i):
        #        with open(f'{str(numi)}.txt', 'a') as f:
        #            f.write(j)
        return self.scf_converged_blocks

    @property
    def get_no_atoms(self) -> int:
        with open(self.output_filepath, 'r') as f:
            lines = f.readlines()
            for i in lines:
                if self.patterns['NO_ATOMS']['pattern'] in i:
                    no_atoms = int(i.split()[5])
        return no_atoms 

    # REVIEW: property decorator on scf_converged_blocks does same function
    #def get_scf_blocks(self):
    #    return self.scf_converged_blocks
    
    @property
    def get_number_of_scf_blocks(self) -> int:
        return len(self.set_scf_blocks)

    '''
        AppOutput Collation Methods
    '''
    @property
    def get_geo(self) -> np.ndarray:
        marker = 'Updated atomic structure:'
        marker_final_sp = 'Final atomic structure:'
        start_index = None
        self.geo = np.zeros((self.get_number_of_scf_blocks, self.get_no_atoms, 3))
        cnt = 0
        for numi, i in enumerate(self.set_scf_blocks):
            for numj, j in enumerate(i):
                if marker in j:
                    start_index = numj + 2
                elif start_index is not None and j.strip() == '':
                    end_index = numj - 3
                    atomic_structure = i[start_index: end_index]
                    for numk, k in enumerate(atomic_structure):
                        numbers = [float(x) for x in k.split()[1:4]]
                        self.geo[cnt, numk] = numbers
                    start_index = None
                    cnt += 1

                elif marker_final_sp in j:
                    atomic_structure = i[numj + 2: numj + self.get_no_atoms+2]
                    for numk, k in enumerate(atomic_structure):
                        numbers = [float(x) for x in k.split()[1:4]]
                        self.geo[cnt, numk] = numbers
                    start_index = None
                    cnt += 1
        return self.geo

    @property 
    def get_forces(self) -> np.ndarray:
        marker = 'Total atomic forces (unitary forces cleaned) [eV/Ang]'
        start_index = None
        self.forces = np.zeros((self.get_number_of_scf_blocks, self.get_no_atoms, 3))
        cnt = 0
        for numi, i in enumerate(self.set_scf_blocks):
            for numj, j in enumerate(i):
                if marker in j:
                    start_index = numj + 1
                elif start_index is not None and j.strip() == '':
                    end_index = numj
                    force = i[start_index:end_index]
                    for numk, k in enumerate(force):
                        numbers = [float(x) for x in k.split()[-3:]]
                        self.forces[cnt, numk] = numbers
                    start_index = None
                    cnt += 1
        return self.forces 


    @property
    def get_vib_eigvec(self) -> np.ndarray:
        check_vib = [x for x in os.listdir('./') if 'vibration' in x][0]
        if len(check_vib) == 0:
            raise ValueError("Cannot find 'vibration' directory")
        else: pass

        vib_xyz = [os.path.join(check_vib[0], x) for x in os.listdir('./') if '.xyz' in x][0]
        with open(vib_xyz, 'r') as f:
            lines = f.readlines()
        self.eigvec = np.zeros((self.get_no_atoms*3, self.get_no_atoms, 3))
        for numi, i in enumerate(lines):
            if 'frequency' in i:
                start_index = numi+1
                end_index = numi+1 + self.get_no_atoms
                contents = i[start_index: end_index]
                for numj, j in enumerate(contetns):
                    numbers = [float(x) for x in j.split()[4:]]
                    self.eigvec[numj] = numbers
        return self.eigvec                    



    def get_total_energy(self,block=-1):
        pattern_str = self.patterns['SCF_ENERGY']['pattern'].replace("'","")
        token = int(self.patterns['SCF_ENERGY']['token']) - 1
        pattern = re.compile(pattern_str)
    
        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target = float(line.split()[ token ])
                break
        return target

    def get_dipole(self,block=-1):
        pattern_str = self.patterns['DIPOLE']['pattern'].replace("'","")
        token = int(self.patterns['DIPOLE']['token']) - 1
        pattern = re.compile(pattern_str)       

        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target = float(line.split()[ token ])
                break
        return target

    def get_dipole_moment(self,block=-1):
        pattern_str = self.patterns['DIPOLE_MOMENT']['pattern'].replace("'","")
        token_x = int(self.patterns['DIPOLE_MOMENT']['token_x']) - 1
        token_y = int(self.patterns['DIPOLE_MOMENT']['token_y']) - 1
        token_z = int(self.patterns['DIPOLE_MOMENT']['token_z']) - 1
        pattern = re.compile(pattern_str)

        target = []

        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target.append( float(line.split()[ token_x ]) )
                target.append( float(line.split()[ token_y ]) )
                target.append( float(line.split()[ token_z ]) )
                break
        return target

    def get_homolumo(self,block=-1):

        '''
            field: (0) HOMO (1) LUMO (2) HOMO-LUMO
        '''     
        target = []

        # HOMO
        pattern_str = self.patterns['HOMO']['pattern'].replace("'","")
        token = int(self.patterns['HOMO']['token']) - 1
        pattern = re.compile(pattern_str)
        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target.append( float(line.split()[ token ]) )
                break
        # LUMO
        pattern_str = self.patterns['LUMO']['pattern'].replace("'","")
        token = int(self.patterns['LUMO']['token']) - 1
        pattern = re.compile(pattern_str)
        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target.append( float(line.split()[ token ]) )
                break
        # HOMOLUMO GAP
        pattern_str = self.patterns['HOMOLUMO']['pattern'].replace("'","")
        token = int(self.patterns['HOMOLUMO']['token']) - 1
        pattern = re.compile(pattern_str)
        for line in self.scf_converged_blocks[block]:
            matching = pattern.search(line)
            if matching:
                target.append( float(line.split()[ token ]) )
                break
        return target


    # Getters Miscs

    def get_patterns(self):
        # return type 'json'
        return self.patterns

    def get_tag(self):
        return self.tag




if __name__ == '__main__':

    file_root = '/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_1'
    main_output = file_root + '/FHIaims.out'
    input_geo   = file_root + '/geometry.in'
    output_geo  = file_root + '/geometry.in.next_step'
    
    ext2 = extractor()
    ext2.set_output_filepath(main_output)
    ext2.set_input_geometry_filepath(input_geo)
    ext2.set_output_geometry_filepath(output_geo)
    
    print('check filepaths()')
    print(ext2.check_filepaths())   # if None in ext2.check_filepaths():
    print('calculation success check: {}'.format(ext2.check_calculation_success()))
    
    print('calculation runtime')
    rtime = ext2.check_calculation_runtime()
    print(rtime)

    print('calculation parallel tasks')
    ptask = ext2.check_parallel_task()
    print(ptask)







    ### EXTRACTION

    ext2.set_scf_blocks()       # load scf blocks

    # Energy Check
    print('init E')
    init_E  = ext2.get_total_energy(0)
    print(init_E)
    print('final E')
    final_E = ext2.get_total_energy()
    print(final_E)

    # Dipole Check
    print('init P')
    init_p = ext2.get_dipole(0)
    print(init_p)
    initial_p_elem = ext2.get_dipole_moment(0)
    print(initial_p_elem)

    print('final P')
    final_p = ext2.get_dipole()
    print(final_p)
    final_p_elem = ext2.get_dipole_moment(0)
    print(final_p_elem)

    # HOMOLUMO CHECK
    print('init homo-lumo, list [homo,lumo,homo-lumo]')
    init_hl = ext2.get_homolumo(0)
    print(init_hl)
    print('final homo-lumo, list [homo,lumo,homo-lumo]')
    final_hl = ext2.get_homolumo()
    print(final_hl)

    '''
        Unit test with 'ext2' instance
    '''
    print('--- input')
    ext2.input_geometry.show_info()
    print('--- output')
    ext2.output_geometry.show_info()
    print('in - out geometry rmsd')
    rmsd = calculate_rmsd_molecules(ext2.input_geometry,ext2.output_geometry)
    print(rmsd)

    #print('output check --')
    #ext2.check_output_success()
    #print(ext2.output_success_tag)

    '''
        Unit test getting SCF Blocks
    '''
