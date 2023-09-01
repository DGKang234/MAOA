import sys
from ase import io
from ase.optimize import MDMin
from ase.dyneb import DyNEB
from ase.optimize import BFGS
from mace.calculators import MACECalculator

# Read initial and final states:
start = sys.argv[1]
final = sys.argv[2]
model_to_path = sys.argv[3]
DEVICE = sys.argv[4]
if DEVICE == None:
    DEVICE == 'gpu'
else: pass

initial = io.read(start)
final = io.read(final)
nimages=13

# Make a band consisting of 5 images:
images = [initial]
images += [initial.copy() for i in range(nimages)]
images += [final]

#neb = NEB(images,climb=True)
neb = DyNEB(images, fmax=0.05, dynamic_relaxation=True)

# Interpolate linearly the potisions of the three middle images:
neb.interpolate()
calcs=[ MACECalculator(model_path=model_to_path, device=DEVICE) for i in range(nimages) ]

# Set calculators:
for i in range(nimages):
    images[i+1].calc = calcs[i]

# Optimize:
optimizer = BFGS(neb, trajectory='NEB.traj')
optimizer.run(fmax=0.05)

io.write("traj.xyz", images, write_info=False)

