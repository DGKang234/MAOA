#!/bin/bash -l
# Batch script to run an MPI parallel job with the upgraded software
# stack under SGE with Intel MPI.
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=4:0:0
# 3. Request 1 gigabyte of RAM per process.
#$ -l mem=1G
# 4. Request 15 gigabyte of TMPDIR space per node (default is 10 GB)
# 5. Set the name of the job.
#$ -N N_006
# 6. Select the MPI parallel environment and 24 processes.
#$ -pe mpi 24
# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /scratch/scratch/uccatka/work/AlF3/clusters/rq/n3/n3_rq/ranked/006

# 8. Set the budget
#$ -P Free
#$ -A UCL_chemM_Woodley

# 9. output file
#$ -o /scratch/scratch/uccatka/work/AlF3/clusters/rq/n3/n3_rq/ranked/006
#$ -e /scratch/scratch/uccatka/work/AlF3/clusters/rq/n3/n3_rq/ranked/006

# 10. Run our MPI job.  GERun is a wrapper that launches MPI jobs on our clusters.
gerun /home/uccatka/software/FHIaims-master/build/aims.210914.scalapack.mpi.x > aims.out 
