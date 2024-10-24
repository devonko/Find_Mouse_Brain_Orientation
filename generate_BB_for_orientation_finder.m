%% Generates a NIFTI for use as reference for all images ingested by the orientation finder.
% Need to make sure a NIFTI package is in the path.
addpath('/Users/rja20/Documents/MATLAB/Compressed_sensing_recon/NIFTI_20140122_v2');

% Set an appropriate directory for the output.
output_dir='/Users/rja20';

FOV=20
array_size=96   ;
dims=[array_size, array_size, array_size];
vox=FOV/array_size;
voxel_size=[vox,vox,vox];
origin=-1*(ceil(array_size/2)-1);

% Create a cube with value = 1 in center of image, occupying 1/frac in each dimension.
frac = 4;
starters = ceil(dims*(1/2-1/(frac*2)));
enders = starters + round(dims/frac);
img=zeros(dims);
img(starters(1):enders(1),starters(2):enders(2),starters(3):enders(3))=1;
nii = make_nii(img, [voxel_size], [origin],2);
file=[output_dir '/BB_for_orientation_finder_' num2str(array_size) '.nii.gz'];

save_nii(nii,file)