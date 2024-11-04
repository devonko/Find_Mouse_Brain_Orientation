#! /bin/sh 
#### created by Devon Overson, Nov 2024, email at devonko@gmail.com
# qsub -l h_vmem=64G predict_orientation.sh
#$ -cwd
#$ -o $JOB_NAME.$JOB_ID.out
#$ -j y
#$ -m ea
#$ -q users.q
#$ -wd /log/output/dir/
#$ -M email@email.com

#### use case: qsub -l h_vmem=64G predict_orientation.sh M22090514_dwi_mask_OF_prepped_AIL.nii.gz

exam=$1
parent_path=/path/to/script 
echo "Writing tmp directory..."
mkdir $parent_path/tmp/
sed "s/#exam_id = ###input_exam###/exam_id = [\"$exam\"]/g" $parent_path/tf_orientation_template.py > $parent_path/tmp/tf_orientation_temp.py
python $parent_path/tmp/tf_orientation_temp.py
echo "Cleaning tmp directory..."
rm -r  $parent_path/tmp/




