%% This script is for creating some "fake" brain mask data to test on 
%%%% created by Devon Overson, Nov 2024, email at devonko@gmail.com

close all
clear all

path1 = ['/path/to/github/test_masks'];

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
    target_dimensions = [32,32,32];
    resampled_mask = imresize3(cropped_mask, target_dimensions, 'nearest');
    
    % if mod(k, 2) ~= 0
    %     % rotated_mask_z = rot90(resampled_mask, 1, [1 2]);
    %     rotated_mask_z = pagetranspose(resampled_mask);
    %     classification_1 = 1;
    % else
    %     rotated_mask_z = resampled_mask;
    %     classification_1 = 0;
    % end
    % classifier_array(k,1) = classification_1;
    
    niftiwrite(resampled_mask,[char(path1) '/resampled_masks/' char(fileList2(k))],"Compressed",true);

end


%%





