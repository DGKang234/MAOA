import os
import subprocess
import argparse
from ase import io
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator
import numpy as np
from ase import Atoms
import plotly.graph_objects as go


"""
[For training only:]
python script_name.py --model_path "MACE_n6_test_model" --training_data_path "./Training_set_test.xyz" --testing_data_path "./Testing_set_test.xyz" --validation_data_path "./Validation_set_test.xyz" --radius 6.0 --epochs 100 --device_type "cuda" --layers "128x0e+128x1o+128x2e"


[For training and then single point calculation, you can use:]
python script_name.py --model_path "MACE_n6_test_model" --training_data_path "./Training_set_test.xyz" --testing_data_path "./Testing_set_test.xyz" --validation_data_path "./Validation_set_test.xyz" --radius 6.0 --epochs 100 --device_type "cuda" --layers "128x0e+128x1o+128x2e" --target_stru "path_to_structure_file" --model_path "path_to_model_file" --opt_struc_path "path_to_optimized_structure_file"


[For training and then optimization, you can use:]
python script_name.py --model_path "MACE_n6_test_model" --training_data_path "./Training_set_test.xyz" --testing_data_path "./Testing_set_test.xyz" --validation_data_path "./Validation_set_test.xyz" --radius 6.0 --epochs 100 --device_type "cuda" --layers "128x0e+128x1o+128x2e" --target_stru "path_to_structure_file" --model_path "path_to_model_file" --opt_struc_path "path_to_optimized_structure_file" --optimize


[For only single point calculation (without training):]
python script_name.py --target_stru "path_to_structure_file" --model_path "path_to_model_file" --opt_struc_path "path_to_optimized_structure_file" --skip_training


[For only optimization (without training):]
python script_name.py --target_stru "path_to_structure_file" --model_path "path_to_model_file" --opt_struc_path "path_to_optimized_structure_file" --optimize --skip_training
"""


class MACE:
    def __init__(self, args):
        self.args = args

    def MACE_training(self):
        layers = 'x'.join(self.args.layers.split(' '))  # remove spaces around '+'
        subprocess.check_call([
            "python", "/home/uccatka/software/mace/scripts/run_train.py",
            "--name", self.args.model_path,
            "--train_file", self.args.training_data_path,
            "--test_file", self.args.testing_data_path,
            "--valid_file", self.args.validation_data_path,
            "--config_type_weights", '{"Default":1.0}',
            "--model", "MACE",
            #"--E0s", "{9:0.000, 13:0.000}",                        # for the only IP data
            #"--E0s", "{9:-2707.428895973, 13:-6596.914328816}",     # for the PBEsol data
            "--E0s", "{9:-2711.537676517, 13:-6543.933824960}",
            "--hidden_irreps", layers,
            "--r_max", str(self.args.radius),
            "--batch_size", str(self.args.batch_size),
            "--max_num_epochs", str(self.args.epochs),
            "--swa",
            "--start_swa", "10",
            "--ema",
            "--ema_decay", "0.99",
            "--amsgrad",
            "--restart_latest",
            "--device", self.args.device_type
        ])

    def calculation(self):
        if not (self.args.target_stru and self.args.model_path):
            print("Target structure and model path must be specified for calculation")
            return

        if self.args.optimize:
            initial = io.read(self.args.target_stru)
            initial.set_calculator(MACECalculator(model_path=f"{self.args.model_path}.model", device=self.args.device_type))
            self.optimize(initial, self.args.opt_struc_path)
        else:
            self.single_point(self.args.target_stru, f"{self.args.model_path}.model", self.args.device_type)

        if self.args.vibration == 'y':
            print("\n" * 2)
            opt_struc_vib = io.read(self.args.opt_struc_path)
            vib = Vibrations(opt_struc_vib)
            vib.run()
            vib.summary()


    def optimize(self, initial, opt_struc):
        dyn = BFGS(initial, trajectory=f'{opt_struc}_dummy.traj', logfile=f'{opt_struc}.log')
        dyn.run(fmax=0.001)
        #io.write(f"{opt_struc}.xyz", initial, format="xyz")

        read_traj = io.read(f'{opt_struc}_dummy.traj', index=":")
        io.write(f'{opt_struc}_traj.xyz', read_traj, format='extxyz')
        opt_trajectory = f'{opt_struc}_traj.xyz'

        read_opt = io.read(f'{opt_struc}_dummy.traj', index="-1:")
        io.write(f'{opt_struc}.xyz', read_opt, format='extxyz')
        return opt_struc, opt_trajectory


    def single_point(self, target_stru, model_path, device_type):
        structure = io.read(target_stru)
        calculator = MACECalculator(model_path=f"{model_path}.model", device=device_type)
        structure.set_calculator(calculator)
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        return energy, forces


    def dimer_curve(self, model_path, device_type, atom1='Al', atom2='F', distance_range=(0.0, 5.0), num_points=51):
        # List of distances to calculate energies for
        distances = np.linspace(*distance_range, num_points)
        energies_cat_an = []
        energies_an_an = []
        energies_cat_cat = []
        BM_values = []
        buck4_values = []
        print("single point calculation of dimers at range of distance")
        for d in distances:
            d = round(d, 2)
            print(f'{d} Ang')
            # cation-anion interaction (MACE)
            cat_an_dimer = Atoms(f'{atom1}{atom2}', positions=[(0, 0, 0), (0, 0, d)])
            calculator = MACECalculator(model_path=f"{model_path}.model", device=device_type)
            cat_an_dimer.set_calculator(calculator)
            energy_cat_an = cat_an_dimer.get_potential_energy()
            energies_cat_an.append(energy_cat_an)
 
            # anion-anion interaction (MACE) 
            an_an_dimer = Atoms(f'{atom2}{atom2}', positions=[(0, 0, 0), (0, 0, d)])
            #calculator = MACECalculator(model_path=f"{model_path}.model", device=device_type)
            an_an_dimer.set_calculator(calculator)
            energy_an_an = an_an_dimer.get_potential_energy()
            energies_an_an.append(energy_an_an)
            
            # cation-cation interaction (MACE)
            cat_cat_dimer = Atoms(f'{atom1}{atom1}', positions=[(0, 0, 0), (0, 0, d)])
            #calculator = MACECalculator(model_path=f"{model_path}.model", device=device_type)
            cat_cat_dimer.set_calculator(calculator)
            energy_cat_cat = cat_cat_dimer.get_potential_energy()
            energies_cat_cat.append(energy_cat_cat)

            # Calculate Al-F BM value for the current distance
            BM_value = self.BM(d)
            BM_values.append(BM_value)
    
            buck4_value = self.buck4(d)
            buck4_values.append(buck4_value)

        #
        # Getting interatomic distances
        print("Retrieve training data")
        with open('Training_set_test.xyz', 'r') as f:
            full_lines = f.readlines()

        no_atoms = full_lines[0]
        check_continue, cluster_set, clusters, ID, ID_set = [], [], [], [], []
        for numi, i in enumerate(full_lines):
            if len(i) > 10 and "Properties" not in i:
                check_continue.append(numi)
                if numi - check_continue[-1] == 0:
                    clusters.append(i.split()[1:4])
                    ID.append(i.split()[0])
                else: pass

            else:
                if len(clusters) != 0:
                    clusters = np.array(clusters).astype(float)
                    cluster_set.append(clusters) # make nested list
                    ID_set.append(ID)            # same here
                else: pass
                ID, clusters = [], []

        cluster_set = np.array(cluster_set[:-1])

        cat_cat_dist, an_an_dist, cat_an_dist = [], [], []
        print("Calculating ineteratomic distances")
        for i in range(len(cluster_set)):
            (npairs_all, npairs_cat_cat, npairs_an_an, npairs_cat_an, all_dist, c_c_dist, a_a_dist, c_a_dist) = \
                self.RDF(no_atoms, cluster_set[i], ID_set[i])

            cat_cat_dist += c_c_dist # concatenate the list
            an_an_dist += a_a_dist
            cat_an_dist += c_a_dist


        #
        # PLOT
        #
        print("Start plotting")
        BM_color = "rgb(10, 120, 24)"

        fig = go.FigureWidget()
        
        energy_trace_cat_an = fig.add_scatter(
            x=distances,
            y=energies_cat_an,
            mode='lines',
            name='MACE: cation - anion',
            line=dict(shape='linear', color=BM_color)
            )

        energy_trace_an_an = fig.add_scatter(
            x=distances, 
            y=energies_an_an, 
            mode='lines', 
            name='MACE: anion - anion', 
            line=dict(shape="linear", color="firebrick")
            )
        
        energy_trace_cat_cat = fig.add_scatter(
            x=distances, 
            y=energies_cat_cat, 
            mode='lines', 
            name='MACE: cation - cation', 
            line=dict(shape="linear", color="blue")
            )

        BM_trace = fig.add_scatter(
            x=distances, 
            y=BM_values, 
            mode='lines', 
            name='Al-F Born-Mayer', 
            line=dict(color=BM_color, dash="dot")
            )

        buck4_trace = fig.add_scatter(
            x=distances, 
            y=buck4_values, 
            mode='lines', 
            name='F-F Buck4', 
            line=dict(color="firebrick", dash="dot")
            )


        #### upper panel (hover)
        cat_an_dist_trace = fig.add_histogram(
            x=cat_an_dist, #all_het_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color=BM_color,
            name="cat-an dist",
            yaxis="y2")

        cat_cat_dist_trace = fig.add_histogram(
            x=cat_cat_dist, #all_homo_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color="blue",
            name="cat-cat dist",
            yaxis="y2")

        an_an_dist_trace = fig.add_histogram(
            x=an_an_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color="firebrick",
            name="an-an dist",
            yaxis="y2")


        #
        ### Settings/layout ###
        #
        fig.layout = dict(
            xaxis=dict(
                domain=[0, 0.8],
                range=[0,6.0],
                showgrid=False,
                zeroline=False,
                title="Interatomic distance / Å"),
            yaxis=dict(
                domain=[0, 0.8],
                range=[-20, 50],
                showgrid=False,
                zeroline=True,
                title="Potential energy / eV"),
            legend=dict(
                x=0.85,
                y=1.0,
                ),
            margin=dict(l=80, r=80, t=80, b=80),
            width=1400,
            height=800,
            hovermode="closest",
            bargap=0.8,
            font=dict(size=20),
            xaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False),

            # hover plot
            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, title="Count")
        )


        fig.write_html('./dimer_curve.html')
        print("A dimer curve figure is saved")
        fig.show()


    def RDF(self, no_of_atoms, coord, ID):
        # Calculate the interatomic disntaces
        all_dist = []
        c_a_dist = []
        c_c_dist = []
        a_a_dist = []
        npairs_all = 0
        npairs_cat_an = 0
        npairs_cat_cat = 0
        npairs_an_an = 0

        all_dup_filter = []
        cat_an_dup_filter = []
        cat_cat_dup_filter = []
        an_an_dup_filter = []
        for i in range(len(ID)):
            for j in range(i+1, len(ID)):
                npairs_all += 1
                distance = np.linalg.norm(coord[i, :] - coord[j, :])
                all_dist.append(distance)

                # Interatomic distance between hetero species (cat-an)
                if ID[i] != ID[j] and (str(i)+str(j) not in cat_an_dup_filter):
                    npairs_cat_an += 1
                    distance = np.round(np.linalg.norm(coord[i, :] - coord[j, :]), 9)
                    c_a_dist.append(distance)
                    cat_an_dup_filter.append(str(i)+str(j))

                # Interatomic distance between homo species
                anions_list = ['F', 'Cl', 'Br'] # temporal
                if ID[i] in anions_list:
                    if ID[i] == ID[j] and i != j and (str(j)+str(i) not in an_an_dup_filter):
                        npairs_an_an += 1
                        distance = np.round(np.linalg.norm(coord[i,:] - coord[j, :]), 9)
                        a_a_dist.append(distance)
                        an_an_dup_filter.append(str(i)+str(j))

                else:
                    if ID[i] == ID[j] and i != j and (str(j) + str(i) not in cat_cat_dup_filter):
                        npairs_cat_cat += 1
                        distance = np.round(np.linalg.norm(coord[i,:] - coord[j, :]), 9)
                        c_c_dist.append(distance)
                        cat_cat_dup_filter.append(str(i) + str(j))

            return (npairs_all, npairs_cat_cat, npairs_an_an, npairs_cat_an, all_dist, c_c_dist, a_a_dist, c_a_dist)


    #
    # analytical potentials
    #
    def BM(self, x):
        return 3760 * np.exp(-x / 0.222)
    

    def buck4(self, x):  # 2.73154 Å F-F distance
        if x.all() < 2.0:
            return 1127.7 * np.exp(-x / 0.2753)
        elif 2.0 <= x.all() < 2.726:
            return (
                -3.976 * x**5
                + 49.0486 * x**4
                - 241.8573 * x**3
                + 597.2668 * x**2
                - 741.117 * x
                + 371.2706
            )
        elif 2.726 <= x.all() < 3.031:
            return -0.361 * x**3 + 3.2362 * x**2 - 9.6271 * x + 9.4816
        elif x.all() >= 3.031:
            return -15.83 / x**6


    def coulomb(self, r, cat_q, an_q):
        return (cat_q * an_q) / r * 14.3996439067522

    def coulomb_force(self, r, unit_r, cat_q, an_q):
        return (cat_q * an_q) / r**2 * unit_r * 14.3996439067522

    def coulomb_energy(self, structure):
        coulomb_e = 0.0
        forces = np.zeros_like(self.positions)
        
        for i in range(len(self.positions)):
            for j in range(i+1, len(self.positions)):
                coord1 = self.positions[i]
                coord2 = self.positions[j]
                atom1 = self.species[i]
                atom2 = self.species[j]
                
                # Calculate the distance and unit vector between the two atoms
                r_vec = coord2 - coord1
                r = np.linalg.norm(r_vec)
                unit_r = r_vec / r
                
                # Calculate the Coulomb potential energy between the pair
                energy_pair = self.coulomb_energy(r, self.charges[atom1], self.charges[atom2])
                
                # Calculate the Coulomb force between the pair
                force_pair = self.coulomb_force(r, unit_r, self.charges[atom1], self.charges[atom2])
                
                # Add to the total energy
                coulomb_e += energy_pair
                
                # Add forces to atoms
                forces[i] -= force_pair
                forces[j] += force_pair
                
        return coulomb_e, forces





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--model_path', default="MACE_model", help='Name of the model')
    parser.add_argument('--training_data_path', default="./Training_set_test.xyz", help='Path to training data')
    parser.add_argument('--testing_data_path', default="./Testing_set_test.xyz", help='Path to testing data')
    parser.add_argument('--validation_data_path', default="./Validation_set_test.xyz", help='Path to validation data')
    parser.add_argument('--radius', type=float, default=6.0, help='Radius (default = 6.0)')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size (default = 5)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--device_type', default="gpu", help='Type of device')
    parser.add_argument('--layers', default="128x0e+128x1o+128x2e", help='Layers')

    parser.add_argument('--target_stru', type=str, help="Path to the structure file")
    parser.add_argument('--opt_struc_path', type=str, help="Path to the optimized structure file")
    parser.add_argument('--vibration', type=str, default='n', help="Perform vibration? (y/n)")
    parser.add_argument('--optimize', action='store_true', help="Perform structure optimization")

    parser.add_argument('--dimer', action='store_true', help="Calculate a dimer curve")

    parser.add_argument('--skip_training', action='store_true', help="Skip training and perform only calculation")

    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # environment variable setting

    mace = MACE(args)
    if not args.skip_training:
        mace.MACE_training()
    if args.target_stru and args.model_path:
        mace.calculation()

    if args.dimer:
        if args.model_path:
            mace.dimer_curve(args.model_path, args.device_type)
        else:
            print("Model path must be specified to calculate a dimer curve")

