import sys
import os
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

def single_point(self, target_stru, model_path, device):
    structure = io.read(target_stru)
    calculator = MACECalculator(model_path=model_path, device=device)
    structure.set_calculator(calculator)
    energy = structure.get_potential_energy()
    forces = structure.get_forces()
    return energy, forces


if sys.argv[1] == '-h':
    print()
    print("Instruction:")
    print("python geoopt.py {struc_path} {model_path} {opt_struc} {vibration} {DEVICE}")
else: pass


#try:
struc_path = sys.argv[1]
model_path = sys.argv[2]
opt_struc = sys.argv[3]
vibration = sys.argv[4].lower()

if vibration == None:
    vibration == 'n'
else: pass

DEVICE = sys.argv[5]
if DEVICE == None:
    DEVCIE = 'gpu'
else: pass

initial = io.read(struc_path)
initial.set_calculator(MACECalculator(model_path=model_path, device=DEVICE))
#optimize(initial, opt_struc)
energy, force = single_point(struc_path, model_path, device)
print(energy, forces)


if vibration == 'y':
    print()
    print()
    opt_struc_vib = io.read(opt_struc)
    vib = Vibrations(opt_struc_vib)
    vib.run()
    vib.summary()
else: pass

#except IndexError or NameError:
#    print("Error")
#    print("python geoopt.py {struc path want to opt} {model path} {name for opt struc} {vibration (default n)} {device (default gpu)}")
#    print()


