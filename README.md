# MAOA
#### Contributor: Dr. Woongkyu Jee, Dong-Gi Kang
* * *
#### This repository contains scripts that can take essential data from atomic simulation calculations  </br>
### Software that can work with: FHI-aims, GULP

It has functions to take data from output file: </br>
✅ SCF converged atomic coordination </br>
✅ Forces </br>
✅ Energies </br>
✅ Eigenvector of vibrational mode </br>
✅ Prepare training data for ML-IP using cartesian coordinate, force, and eigenvector of vibrational mode (FHI-aims) </br>
✅ Prepare training data for ML-IP from taking SCF converged cycles in optimisation process (FHI-aims) </br>



☉ Upcoming function: </br>
  **1.** Take top _n_ many local minima of KLMC output (top_structure) to make FHI-aims calculation directories (geometry.in, control.in, submit.sh) </br>
  **2.** preparing vibration calculation, and submit the job (SGE) </br>

## Log </br>
✅ May 2023: geometry, forces, energy, eigenvector of vibration mode (and training data), training data using SCF converged cycles. </br>
