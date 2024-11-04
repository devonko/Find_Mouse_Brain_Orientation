#### template for orientation prediction
#### created by Devon Overson, Nov 2024, email at devonko@gmail.com

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

path1=["path/to/files"]
#### load model
model_id=[8581]
model = load_model(str(path1[0])+"/od1_model_rev1_run"+str(model_id[0])+".keras")
model.summary()

#### load dictionary
label_df = pd.read_csv(str(path1[0])+"/label_mapping.csv")  
label_dict = pd.Series(label_df.name.values, index=label_df.label).to_dict()

### predict exam
#exam_id = ###input_exam###

im_test=[]
nifti_file = [str(path1[0])+"/github/test_masks/"+str(exam_id[0])] # Folder containing your .nii.gz files
img1 = nib.load(str(nifti_file[0]))
mask = img1.get_fdata()

rows, cols, slices = np.where(mask)
row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()
slice_min, slice_max = slices.min(), slices.max()
# Crop the mask
cropped_mask = mask[row_min:row_max+1, col_min:col_max+1, slice_min:slice_max+1]
# Define target dimensions
target_dimensions = (32, 32, 32)
# Calculate zoom factors
zoom_factors = [t / s for t, s in zip(target_dimensions, cropped_mask.shape)]
# Resample the cropped mask
resampled_mask = zoom(cropped_mask, zoom_factors, order=0)  # nearest-neighbor interpolation

# im_test.append = np.array(resampled_mask)
im_test.append(resampled_mask)
im_test = np.expand_dims(im_test, axis=-1)  # Add channel dimension -> (depth, height, width, 1)

#### evaluate image with model:
CN_pred = model.predict(im_test,verbose=0)
# print((CN_pred))
predicted_class = np.argmax(CN_pred)  # Use np.argmax to get the class index if CN_pred is a probability distribution
print("File: "+str(exam_id[0])+" Predicted class:", label_dict.get(predicted_class, "Unknown"))

output_df = pd.DataFrame({
    'file': exam_id,
    'file_path': nifti_file, 
    'predicted_class': predicted_class,
    'predicted_class2': label_dict.get(predicted_class, "Unknown")
})

# Export to CSV
output_df.to_csv(str(path1[0])+"/prediction_report.csv", index=False)
print("Predictions saved to predictions_report")


