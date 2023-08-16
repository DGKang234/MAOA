from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor
import os
from itertools import groupby
import numpy as np

eigvec_path = [os.path.join('sp', x) for x in os.listdir('sp')]
sp_path = [os.path.join(dirpath, fname) for dirpath in eigvec_path for fname in os.listdir(dirpath)]
lambda_path = [os.path.join(dirpath, fname) for dirpath in sp_path for fname in os.listdir(dirpath) if fname == 'aims.out']
aims_out_path = sorted(lambda_path, key=lambda x: (int(x.split('/')[1]), float(x.split('/')[2].split('_')[1])))
aims_out_path = [list(group) for key, group in groupby(aims_out_path, lambda x: int(x.split('/')[1]))]
ex = extractor()
if not os.path.exists('ext_xyz'):
    os.mkdir('ext_xyz')

for numi, i in enumerate(aims_out_path):
    total_energy = []
    geometry = []

    for j in i:
        print(j)
        ex.set_output_filepath(j)
        ex.set_scf_blocks

        ex.get_no_atoms
        
        ex.get_sp_geometries(j)
        ex.get_sp_atom_order()
        ex.get_sp_species()
        ex.get_forces
        force_shape = np.shape(ex.get_forces)
        get_forces = np.reshape(ex.get_forces, (force_shape[1], force_shape[2]))

        form = np.concatenate((ex.get_sp_atom_order(), ex.get_sp_geometries(j), get_forces), axis=1)

        total_energy.append(ex.get_total_energy())
        geometry.append(form)

    for numk, k in enumerate(total_energy):
        with open(f"ext_xyz/ext_{j.split('/')[1]}_eigv.xyz", 'a') as f:
            f.write(str(force_shape[1]) + '\n')
            f.write(f'Properties-species:S:1:pos:R:3:forces:R:3 energy={total_energy[numk]} pbc="F F F"\n')
            np.savetxt(f, geometry[numk], fmt="%s", delimiter="    ")


        #ex.get_vib_eigvec
#        for k in range(len(ex.set_scf_blocks)):
#            #print(ex.get_total_energy(k))
#            #print('Geometry')
#            print(ex.get_sp_geometries)
#            #print('Atomic forces')
#            #print(ex.get_forces(k))
#            #print('Eigenvector of vibrational modes')
#            #print(ex.get_vib_eigvec)
#            #print()
#            #print()
#        ex.get_total_energy

