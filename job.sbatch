#!/bin/bash
#SBATCH -J twin-listrik-kampus
#SBATCH -N 1                      # jumlah node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # jumlah core yang diminta
#SBATCH -t 00:30:00               # batas waktu
#SBATCH -p compute                # nama partisi (sesuaikan dengan cluster)
#SBATCH -o slurm-%j.out

# module load Python              # sesuaikan dengan cluster
python hpc_case.py --workers $SLURM_CPUS_PER_TASK --replicates 200
