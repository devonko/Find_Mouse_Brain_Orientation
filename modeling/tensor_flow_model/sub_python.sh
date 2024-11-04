#! /bin/sh 
#### created by Devon Overson, Nov 2024, email at devonko@gmail.com

# Run by command: 
# qsub -l h_vmem=64G sub_python.sh 
#$ -cwd
#$ -o $JOB_NAME.$JOB_ID.out
#$ -j y
#$ -m ea
#$ -q users.q
#$ -wd /log/output/dir/
#$ -M email@email.com
exam=$1
python /path/to/script/${exam}

