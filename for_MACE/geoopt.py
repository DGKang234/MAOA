import sys
import os
import argparse
import subprocess
from ase import io
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator


class MACE:
    def __init__(self):
		pass

    def training_MACE(self):
        ## Define the command-line arguments
        #parser = argparse.ArgumentParser(description="MACE Training Instructions")
        #parser.add_argument('out_model_name', type=str, help="Output model name")
        #parser.add_argument('training_data_path', type=str, help="Path to the training data file")
        #parser.add_argument('testing_data_path', type=str, help="Path to the testing data file")
        #parser.add_argument('radius', type=float, help="Radius")
        #parser.add_argument('epochs', type=int, help="Number of epochs")
        #parser.add_argument('device_type', type=str, default='cpu', help="Device type (default: cpu)")
        #parser.add_argument('layers', type=str, nargs='?', default='128x0e + 128x1o + 129x2e',
        #                    help="NN layers (default: '128x0e + 128x1o + 129x2e')")

        ## Parse the command-line arguments
        #args = parser.parse_args()

        ## Access the parsed arguments
        #out_model_name = args.out_model_name
        #training_data_path = args.training_data_path
        #testing_data_path = args.testing_data_path
        #radius = args.radius
        #epochs = args.epochs
        #device_type = args.device_type
        #layers = args.layers

        # Process the layers argument
        if layers is None:
            layers = '128x0e + 128x1o + 129x2e'

        # [TEMPORARY] Read the training, testing, and validation data
        train = io.read(training_data_path, index=":30")
        test = io.read(training_data_path, index="-10:")
        valid = io.read(training_data_path, index="30:35")

        # Write the training, testing, and validation data to files
        io.write('./FIT/Training_set_test.xyz', train, format='extxyz')
        io.write('./FIT/Testing_set_test.xyz', test, format='extxyz')
        io.write('./FIT/Validation_set_test.xyz', valid, format='extxyz')


    def optimize(self, initial_placer, opt_struc):
        dyn = BFGS(initial_placer, trajectory=f'{opt_struc}_dummy.traj')
        dyn.run(fmax=0.001)
        io.write(opt_struc, initial_placer, format="xyz")

        read_traj = io.read(f'{opt_struc}_dummy.traj', index=":")
        io.write(f'{opt_struc}_traj.xyz', read_traj)
        opt_trajectory = f'{opt_struc}_traj.xyz'
        return opt_struc, opt_trajectory


    def single_point(self, target_stru, model_path, device):
        structure = io.read(target_stru)
        calculator = MACECalculator(model_path=model_path, device=device)
        structure.set_calculator(calculator)
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
        return energy, forces


    def calculation(self):
        #parser = argparse.ArgumentParser(description="Geometric Optimization Instructions")
        #parser.add_argument('target_stru', type=str, help="Path to the structure file")
        #parser.add_argument('model_path', type=str, help="Path to the model file")
        #parser.add_argument('opt_struc_path', type=str, help="Path to the optimized structure file")
        #parser.add_argument('vibration', type=str, default='n', help="Perform vibration? (y/n)")
        #parser.add_argument('device', type=str, default='gpu', help="Device to use for computation (default: gpu)")
        #parser.add_argument('--optimize', action='store_true', help="Perform structure optimization")

        #args = parser.parse_args()

        #target_stru = args.target_stru
        #model_path = args.model_path
        #opt_struc = args.opt_struc_path
        #vibration = args.vibration.lower()
        #device = args.device

        if args.optimize:
            initial = io.read(target_stru)
            initial.set_calculator(MACECalculator(model_path=model_path, device=device))
            self.optimize(initial, opt_struc)
        else:
            self.single_point(target_stru, model_path, device)

        if vibration == 'y':
            print("\n" * 2)
            opt_struc_vib = io.read(opt_struc)
            vib = Vibrations(opt_struc_vib)
            vib.run()
            vib.summary()



if __name__ == '__main__':
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="MACE Training Instructions")
    parser.add_argument('out_model_name', type=str, help="Output model name")
    parser.add_argument('training_data_path', type=str, help="Path to the training data file")
    parser.add_argument('testing_data_path', type=str, help="Path to the testing data file")
    parser.add_argument('radius', type=float, help="Radius")
    parser.add_argument('epochs', type=int, help="Number of epochs")
    parser.add_argument('device_type', type=str, default='cpu', help="Device type (default: cpu)")
    parser.add_argument('layers', type=str, nargs='?', default='128x0e + 128x1o + 129x2e',
                        help="NN layers (default: '128x0e + 128x1o + 129x2e')")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    out_model_name = args.out_model_name
    training_data_path = args.training_data_path
    testing_data_path = args.testing_data_path
    radius = args.radius
    epochs = args.epochs
    device_type = args.device_type
    layers = args.layers

    # Create an instance of the MACE class
    mace = MACE()

    # Perform the MACE training
    mace.training_MACE()



	parser = argparse.ArgumentParser(description="Geometric Optimization Instructions")
	parser.add_argument('target_stru', type=str, help="Path to the structure file")
	parser.add_argument('model_path', type=str, help="Path to the model file")
	parser.add_argument('opt_struc_path', type=str, help="Path to the optimized structure file")
	parser.add_argument('vibration', type=str, default='n', help="Perform vibration? (y/n)")
	parser.add_argument('device', type=str, default='gpu', help="Device to use for computation (default: gpu)")
	parser.add_argument('--optimize', action='store_true', help="Perform structure optimization")
	
	args = parser.parse_args()
	
	target_stru = args.target_stru
	model_path = args.model_path
	opt_struc = args.opt_struc_path
	vibration = args.vibration.lower()
	device = args.device
    # Perform the calculation (optimization or single-point)
    mace.calculation()
