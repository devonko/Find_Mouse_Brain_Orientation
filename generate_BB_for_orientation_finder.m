addpath('/Users/rja20/Documents/MATLAB/Compressed_sensing_recon/NIFTI_20140122_v2');
FOV=20
%array_size=64;
array_size=96   ;
dims=[array_size, array_size, array_size];
vox=FOV/array_size;
%vox=0.3125;
voxel_size=[vox,vox,vox];
%origin=-voxel_size*(ceil(array_size/2)-1);
origin=-1*(ceil(array_size/2)-1);
%origin=[-9.6875,-9.6875,-9.6875];
frac = 4;
starters = ceil(dims*(1/2-1/(frac*2)));
enders = starters + round(dims/frac);
img=zeros(dims);
img(starters(1):enders(1),starters(2):enders(2),starters(3):enders(3))=1;
nii = make_nii(img, [voxel_size], [origin],2);
file=['/Users/rja20/BB_for_orientation_finder_' num2str(array_size) '.nii.gz'];
save_nii(nii,file)