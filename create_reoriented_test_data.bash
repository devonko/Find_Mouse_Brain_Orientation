#! /bin/env bash
image=$1;
output_dir=$2;
if [[ ${image:0:1} != '/' ]];then
        file_name=$image;
        folder=$PWD;
else
        file_name=${image##*/};
        folder=${image%/*}
fi

if [[ ! -d ${output_dir} ]]; then
        mkdir -m 775 ${output_dir};
fi

sbatch_dir=${output_dir}/sbatch;

if [[ ! -d ${sbatch_dir} ]]; then
        mkdir -m 775 ${sbatch_dir};
fi
GD=/mnt/clustertmp/common/rja20_dev/gunnies/;
in_code=ALS;

out_codes=();
RHS=(ALS PRS ARI PLI RAS LPS LAI RPI SAL SPR IAR IPL SRA SLP ILA IRP LSA RSP RIA LIP ASR PSL AIL PIR);
LHS=(PLS ARS PRI ALI LAS RPS RAI LPI IAL IPR SAR SPL IRA ILP SLA SRP RSA LSP LIA RIP PSR ASL PIL AIR);

# Assume things are all right-handed:
out_codes=(${RHS[@]}) 


if ((0));then
	input=${tmp_work}${file_name}_down_sampled.nii.gz;
	ovs_1=$(PrintHeader ${target} 1 | cut -d  'x' -f1);
	ovs_2=$(PrintHeader ${target} 1 | cut -d  'x' -f2);
	ovs_3=$(PrintHeader ${target} 1 | cut -d  'x' -f3);
	
	a=$( bc -l <<<"8*$ovs_1 " )
	b=$( bc -l <<<"8*$ovs_2 " )
	c=$( bc -l <<<"8*$ovs_3 " )
	#a=$(( 8*ovs_1 ));
	#b=$(( 8*ovs_2 ));
	#c=$(( 8*ovs_3 ));
	
	if [[ ! -f ${input} ]];then
			#ResampleImageBySpacing 3 ${image} ${input} .2 .2 .2 0;
			ResampleImageBySpacing 3 ${image} ${input} ${a} ${b} ${c} 0;
	fi
	
	ds_target=${tmp_work}${target_name%.nii???}_x8_downsampled.nii.gz;
	if [[ ! -f ${ds_target} ]];then
			#ResampleImageBySpacing 3 ${image} ${input} .2 .2 .2 0;
			ResampleImageBySpacing 3 ${target_folder}/${target_name} ${ds_target} ${a} ${b} ${c} 0;
	fi
fi

for out_code in ${out_codes[@]}; do
        out_image=${output_dir}/${file_name%.nii???}_${out_code}.nii.gz;
        if [[ ! -f ${out_image} ]]; then
            RO_cmd="bash /mnt/clustertmp/common/rja20_dev//matlab_execs_for_SAMBA//img_transform_executable/run_img_transform_exec.sh /mnt/clustertmp/common/rja20_dev//MATLAB2015b_runtime/v90 ${image} ${in_code} ${out_code} ${out_image};fslmaths ${out_image} -bin ${out_image} -odt 'char'";
			job_name=reorient_${file_name%%_*}_${out_code};
			cmd="${GD}submit_sge_cluster_job.bash ${sbatch_dir} ${job_name} 0 0 ${RO_cmd}";
			echo $cmd;
			$cmd;
        fi
done
