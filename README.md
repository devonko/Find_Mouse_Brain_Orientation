# Find_Mouse_Brain_Orientation
A Deep-Learning model to automatically determine the SPIRAL orientation of a 3D image of a mouse brain.

Input images are normalized to a FOV=20x20x20mm3 bounding box reference image with an array size of either 64x64x64 or 96x96x96.
A translation transform is used to align the center-of-mass of the image within this standardized box.

A MATLAB script, generate_BB_for_orientation_finder.m is included to generate reference images of different FOVs and array sizes. By nature of the problem, it is recommended that this reference is isotropic.

A bash script, prep_image_for_orientation_finder.bash, uses ANTs (Advanced Normalization Toolkit) commands to achieve these image normalization steps. By default, this script ignores any orientation information contained in the NIFTI header of the input images, but will respect such info if a third argument of '0' (zero) is included when calling the script. 
