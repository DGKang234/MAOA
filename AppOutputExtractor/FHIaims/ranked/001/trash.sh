#!/bin/bash -l
# Batch script to run an MPI parallel job with the upgraded software
# stack under SGE with Intel MPI.
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=1:0:0
# 3. Request 1 gigabyte of RAM per process.
#$ -l mem=2G
# 4. Request 15 gigabyte of TMPDIR space per node (default is 10 GB)
# 5. Set the name of the job.
#$ -N n3
# 6. Select the MPI parallel environment and 24 processes.
#$ -pe mpi 40
# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -cwd

# 8. Set the budget
#$ -P Gold
#$ -A UCL_chemM_Woodley

# 9. Send email when the job finishes or aborts.
#$ -m e
#$ -M tonggih.kang.18@ucl.ac.uk

# 10. Run our MPI job.  GERun is a wrapper that launches MPI jobs on our clusters.
gerun /home/uccatka/software/FHIaims-master/build/aims.210914.scalapack.mpi.x > aims.out
