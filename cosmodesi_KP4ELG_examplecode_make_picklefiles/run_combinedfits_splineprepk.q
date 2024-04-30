#!/bin/bash -l
#SBATCH -J runcombinedfits_mocks_Abbe
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH --array=0-25
#SBATCH --ntasks=1
#SBATCH --account=desi
##SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH -t 08:00:00
#SBATCH -o combinedfits_secondgen_mocks_v1_2_reducedcov/output/output_%a.txt
#SBATCH -e combinedfits_secondgen_mocks_v1_2_reducedcov/error/error_%a.txt

IDIR=/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles

conda activate barry_env_desiproject_aw
export executable=$(which python)
echo $executable

export PROG='/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles/runcombinedfits_mocks.py'
echo $PROG

cd $IDIR
srun -n 1 -c 1 $executable $PROG ${SLURM_ARRAY_TASK_ID} spline pre pk

