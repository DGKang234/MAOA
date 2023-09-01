import os
import sys
import argparse
from ase import io
from ase.optimize import MDMin
from ase.dyneb import DyNEB
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons
from mace.calculators import MACECalculator


class MACE:
    def __init__(self):
        pass

    def training_MACE(self, training_data_path, layers):
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

    def calculation(self, target_stru, model_path, opt_struc, vibration, device, optimize):
        if optimize:
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
    
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="MACE [Training] and [Geometric Optimization Instructions]")
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')
    
    # Create the sub-parser for the training command
    parser_train = subparsers.add_parser('training', help='Training-related instructions')
    parser_train.add_argument('--out_model_name', type=str, help="Output model name")
    parser_train.add_argument('--training_data_path', type=str, default="./FIT/Training_set_test.xyz", help="Path to the training data file")
    parser_train.add_argument('--testing_data_path', type=str, default="./FIT/Testing_set_test.xyz", help="Path to the testing data file")
    parser_train.add_argument('--validation_data_path', type=str, default="./FIT/Validation_set_test.xyz", help="Path to validation data (default=./FIT/Validation_set_test.xyz)")
    parser_train.add_argument('--radius', type=float, default="6.0", help="Radius")
    parser_train.add_argument('--epochs', type=int, default="100", help="Number of epochs")
    parser_train.add_argument('--device_type', type=str, default='cpu', help="Device type (default: cpu)")
    parser_train.add_argument('--layers', type=str, nargs='?', default='128x0e + 128x1o + 129x2e', help="NN layers (default: '128x0e + 128x1o + 129x2e')")
    
    # Create the sub-parser for the calculation command
    parser_calc = subparsers.add_parser('calculation', help='Calculation-related instructions')
    parser_calc.add_argument('--optimize', action='store_true', help="Perform structure optimization")
    parser_calc.add_argument('--target_stru', type=str, help="Path to the structure file")
    parser_calc.add_argument('--model_path', type=str, help="Path to the model file")
    parser_calc.add_argument('--opt_struc_path', type=str, help="Path to the optimized structure file")
    parser_calc.add_argument('--vibration', type=str, default='n', help="Perform vibration? (y/n)")
    parser_calc.add_argument('--device', type=str, default='gpu', help="Device to use for computation (default: gpu)")
    
    args = parser.parse_args()
    
    command = args.command
    
    if command == 'training':
        # Access the parsed training arguments
        out_model_name = args.out_model_name
        training_data_path = args.training_data_path
        testing_data_path = args.testing_data_path
        radius = args.radius
        epochs = args.epochs
        device_type = args.device_type
        layers = args.layers
    
        # Perform the training-related instructions...
    
        print("Instructions for Training:")
        print(f"- Output model name: {out_model_name}")
        print(f"- Training data path: {training_data_path}")
        print(f"- Testing data path: {testing_data_path}")
        print(f"- Validation data path: {validation_data_path}")
        print(f"- Radius: {radius}")
        print(f"- Epochs: {epochs}")
        print(f"- Device type: {device_type}")
        print(f"- Layers: {layers}")
    
    elif command == 'calculation':
        # Access the parsed calculation arguments
        optimize = args.optimize
        target_stru = args.target_stru
        model_path = args.model_path
        opt_struc = args.opt_struc_path
        vibration = args.vibration.lower()
        device = args.device
    
        # Perform the calculation-related instructions...
    
        print("Instructions for Calculation:")
        print(f"- Optimize: {optimize}")
        print(f"- Target structure path: {target_stru}")
        print(f"- Model path: {model_path}")
        print(f"- Optimized structure path: {opt_struc}")
        print(f"- Vibration: {vibration}")
        print(f"- Device: {device}")



