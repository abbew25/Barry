#!/bin/bash -l
#SBATCH -J runcombinedfits_mocks_Abbe
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH --array=78-103
#SBATCH --ntasks=1
#SBATCH --account=desi
#SBATCH --cpus-per-task=1
##SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 08:00:00
#SBATCH -o pscratch/sd/a/abbew25/combinedfits_secondgen_mocks_v1_2_shuffled/output/output_%a.txt
#SBATCH -e pscratch/sd/a/abbew25/combinedfits_secondgen_mocks_v1_2_shuffled/error/error_%a.txt

IDIR=/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles

conda activate barry_env_desiproject_aw
export executable=$(which python)
echo $executable

export PROG='/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles/runcombinedfits_mocks-with_shuffling.py'
echo $PROG

cd $IDIR
srun -n 1 -c 1 $executable $PROG ${SLURM_ARRAY_TASK_ID} poly post xi 

