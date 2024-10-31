#### run after s2 is run, this will pull the saved model (specify the ID number below) and report if the mask is in the proper orientation or not
#### ensure the following are downloaded:
## nibabel - nifti image format read
## tensorflow - modeling
## sklearn - helps prepare the input data

##### imports
import nibabel as nib
import pandas as pd
import os # used to read the number of .nii.gz files 
import numpy as np
import random # used to create a .keras model run number
from scipy.ndimage import zoom
import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

path1=["/parent/path"]
#### load model
model_id=[4727]
model = load_model(str(path1[0])+"/s2_model_rev1_run"+str(model_id[0])+".keras")
model.summary()

#### load dictionary, this is an optional line that will allow printing of orientaiton classifcaiton, based on the label mapping csv file. Just be sure the labels correspond to those in the master spreadsheet
label_df = pd.read_csv(str(path1[0])+"/label_mapping.csv")  
label_dict = pd.Series(label_df.name.values, index=label_df.label).to_dict()

### test all exams 
df = pd.read_csv(str(path1[0])+"/orientation_detect_modeling/master_spreadsheet2.csv")
#df = df[df['test_group_classification'] == 1] ## <-- you can set this to test only those subjects that are truly test subjects (not part of train/validate sets)
nifti_files = df['file_name'].tolist()

for k in range(len(nifti_files)):
    im_test=[]
    nifti_file = [str(path1[0])+"/github/test_masks/"+str(nifti_files[k])] # note this pulls the native masks as provided (pre-downsampled). The downsampling is then performed here within the script
    img1 = nib.load(str(nifti_file[0]))
    mask = img1.get_fdata()

    rows, cols, slices = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    slice_min, slice_max = slices.min(), slices.max()
    cropped_mask = mask[row_min:row_max+1, col_min:col_max+1, slice_min:slice_max+1] # mask cropped to size
    target_dimensions = (32, 32, 32) #ensure this is the same as the training/test set
    zoom_factors = [t / s for t, s in zip(target_dimensions, cropped_mask.shape)] # determine zoom factors
    resampled_mask = zoom(cropped_mask, zoom_factors, order=0)  # nearest-neighbor interpolation is order=0-\

    im_test.append(resampled_mask)
    im_test = np.expand_dims(im_test, axis=-1)  # Add channel dimension -> (depth, height, width, 1)

    #### evaluate image with model:
    CN_pred = model.predict(im_test,verbose=0)
    # print((CN_pred))
    predicted_class = np.argmax(CN_pred)  # Use np.argmax to get the class index if CN_pred is a probability distribution
    
    print("File: "+str(nifti_files[k])+" Predicted class:", label_dict.get(predicted_class, "Unknown"))

    # else:
    #     print("Proper orientation")
    #     # print(np.round(CN_pred))
