%% this script is for re-sampling masks
close all
clear all

path1 = ['/parent/path'];

fileStruct = dir([char(path1) '/*.nii.gz']);
fileList = {fileStruct.name}; 
fileList2 = erase(fileList, '.nii.gz');   

%%
for k = 1:length(fileList)
    mask = niftiread([char(path1) '/' char(fileList2(k)) '.nii.gz']);
    [rows, cols, slices] = ind2sub(size(mask), find(mask));
    row_min = min(rows);
    row_max = max(rows);
    col_min = min(cols);
    col_max = max(cols);
    slice_min = min(slices);
    slice_max = max(slices);
    cropped_mask = mask(row_min:row_max, col_min:col_max, slice_min:slice_max);
    target_dimensions = [32,32,32]; %% <--- this can be adjusted to capture better resolution, but it there seems to be an undesireable trade-off b/t resolution and model complexity beyond 32 (e.g., 64)
    resampled_mask = imresize3(cropped_mask, target_dimensions, 'nearest');    
    niftiwrite(resampled_mask,[char(path1) '/resampled_masks/' char(fileList2(k))],"Compressed",true);

end
