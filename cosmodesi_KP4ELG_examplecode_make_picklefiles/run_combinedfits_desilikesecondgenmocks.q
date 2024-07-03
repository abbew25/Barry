#!/bin/bash -l
#SBATCH -J runcombinedfits_mocks_Abbe_desilike
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH --array=0-25
#SBATCH --ntasks=1
#SBATCH --account=desi
##SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH -t 08:00:00
#SBATCH -o combinedfits_secondgen_mocks_v1_2_elgslrgsbaselineDESILIKE/output/output_%a.txt
#SBATCH -e combinedfits_secondgen_mocks_v1_2_elgslrgsbaselineDESILIKE/error/error_%a.txt

IDIR=/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles

conda activate /global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11
export executable=$(which python)
echo $executable

export PROG='/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles/runcombinedfits_mocks-desilike.py'
echo $PROG

cd $IDIR
srun -n 1 -c 1 $executable $PROG ${SLURM_ARRAY_TASK_ID} 

