import os
import argparse
from ase import io
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator

def optimize(initial_placer, opt_struc):
    dyn = BFGS(initial_placer, trajectory='dummy.traj')
    dyn.run(fmax=0.001)
    io.write(opt_struc, initial_placer, format="extxyz")

    read_traj = io.read('dummy.traj', index=":")
    io.write('traj.xyz', read_traj)

def single_point(target_stru, model_path, device):
    structure = io.read(target_stru)
    calculator = MACECalculator(model_path=model_path, device=device)
    structure.set_calculator(calculator)
    energy = structure.get_potential_energy()
    forces = structure.get_forces()
    return energy, forces

def main(args):
    initial = io.read(args.struc_path)
    initial.set_calculator(MACECalculator(model_path=args.model_path, device=args.device))

    #optimize(initial, args.opt_struc)
    energy, force = single_point(args.struc_path, args.model_path, args.device)
    print(energy, forces)

    if args.vibration:
        opt_struc_vib = io.read(args.opt_struc)
        vib = Vibrations(opt_struc_vib)
        vib.run()
        vib.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimization and vibration calculation.')
    parser.add_argument('struc_path', help='Path of the structure to be optimized')
    parser.add_argument('model_path', help='Path of the model')
    parser.add_argument('opt_struc', help='Name for the optimized structure')
    parser.add_argument('-v', '--vibration', action='store_true', help='Perform vibration (default is False)')
    parser.add_argument('-d', '--device', default='gpu', help='Device to use (default is gpu)')

    args = parser.parse_args()
    main(args)

