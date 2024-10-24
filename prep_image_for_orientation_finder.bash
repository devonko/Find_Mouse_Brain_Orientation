#! /bin/env bash

input=$1;
work_dir=$2;
ignore_header=$3;

if [[ "xx" == "x${ignore_header}x" ]];then
	ignore_header=1;
fi


input_file=${input##*/};
input_file=${input_file/\.gz/};
if (($ignore_header));then
	reset_img="${work_dir}/${input_file/\.nii/_reset\.nii\.gz}";
	SetDirectionByMatrix ${input} ${reset_img} -1 0 0 0 -1 0 0 0 1;
else
	reset_img=$input;
fi
prefix="${work_dir}/${input_file/\.nii/}_";
ref=~/BB_for_orientation_finder_96.nii.gz;
output=${prefix}OF_prepped.nii.gz;



antsRegistration -v 1 -d 3  -t Translation[1] -r [${ref},${reset_img},1] -m Mattes[${ref},${reset_img},1,32,None] -c [0,1e-8,20] -f 8 -s 4 -z 0 -o ${prefix};
rm ${prefix}1Translation.mat;

translation=${prefix}0DerivedInitialMovingTranslation.mat;

antsApplyTransforms -v 1 -d 3  -i ${reset_img} -r ${ref} -n Linear  -o ${output} -t ${translation}