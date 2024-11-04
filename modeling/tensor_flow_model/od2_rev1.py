#### created by Devon Overson, Nov 2024, email at devonko@gmail.com
#### run after od1, this will pull the saved model and report if the mask is in the proper orientation or not

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
model_id=[8581] ###<-- load model of your liking
model = load_model(str(path1[0])+"/od1_model_rev1_run"+str(model_id[0])+".keras")
model.summary()

#### load dictionary
label_df = pd.read_csv(str(path1[0])+"/label_mapping.csv")  
label_dict = pd.Series(label_df.name.values, index=label_df.label).to_dict()

### test all exams 

df = pd.read_csv(str(path1[0])+"/master_spreadsheet2.csv")
df = df[df['test_group_classification'] == 1] ## <-- you can set this to test only those subjects that are truly test subjects (not part of train/validate sets)
nifti_files = df['file_name'].tolist()

for k in range(len(nifti_files)):
    im_test=[]
    nifti_file = [str(path1[0])+"/github/test_masks/"+str(nifti_files[k])] # Folder containing your .nii.gz files
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
    
    print("File: "+str(nifti_files[k])+" Predicted class:", label_dict.get(predicted_class, "Unknown"))


    # pred_value = CN_pred[0, 0]  # Get the predicted probability (scalar)
    # if pred_value > 0.5:
    #     print("Not in proper orientation, rotate 90 degrees")
    # else:
    #     print("Proper orientation")
    #     # print(np.round(CN_pred))



# #### junkyard

#### test files alternative

# test_orientations = ["AIL" , "ALS" , "ARI" , "ASR" , "IAR" , "ILA" , "IPL" , "IRP" , "LAI" , "LIP" , "LPS" , "LSA" , "PIR" , "PLI" , "PRS" , "PSL" , "RAS" , "RIA" , "RPI" , "RSP" , "SAL" , "SLP" , "SPR" , "SRA"]
# for k in range(len(test_orientations)):
#     im_test=[]

#     # nifti_file = [str(path1[0])+"/github/resampled_masks/N58217_dwi_mask_OF_prepped_"+str(test_orientations[k])+".nii.gz"] # Folder containing your .nii.gz files
#     nifti_file = [str(path1[0])+"/github/test_masks/N58217_dwi_mask_OF_prepped_"+str(test_orientations[k])+".nii.gz"] # Folder containing your .nii.gz files
#     img1 = nib.load(str(nifti_file[0]))
#     mask = img1.get_fdata()

#     rows, cols, slices = np.where(mask)
#     row_min, row_max = rows.min(), rows.max()
#     col_min, col_max = cols.min(), cols.max()
#     slice_min, slice_max = slices.min(), slices.max()
#     # Crop the mask
#     cropped_mask = mask[row_min:row_max+1, col_min:col_max+1, slice_min:slice_max+1]
#     # Define target dimensions
#     target_dimensions = (32, 32, 32)
#     # Calculate zoom factors
#     zoom_factors = [t / s for t, s in zip(target_dimensions, cropped_mask.shape)]
#     # Resample the cropped mask
#     resampled_mask = zoom(cropped_mask, zoom_factors, order=0)  # nearest-neighbor interpolation

#     # im_test.append = np.array(resampled_mask)
#     im_test.append(resampled_mask)

#     im_test = np.expand_dims(im_test, axis=-1)  # Add channel dimension -> (depth, height, width, 1)



#     #### evaluate image with model:
#     CN_pred = model.predict(im_test,verbose=0)
#     # print((CN_pred))
#     predicted_class = np.argmax(CN_pred)  # Use np.argmax to get the class index if CN_pred is a probability distribution
    
#     print("Ground truth class: "+str(test_orientations[k])+" Predicted class:", label_dict.get(predicted_class, "Unknown"))


#     # pred_value = CN_pred[0, 0]  # Get the predicted probability (scalar)
#     # if pred_value > 0.5:
#     #     print("Not in proper orientation, rotate 90 degrees")
#     # else:
#     #     print("Proper orientation")
#     #     # print(np.round(CN_pred))

#### junkyard

#     nifti_images.append(img.get_fdata())  # Convert NIfTI image to numpy array


# #### separate into train/validate (omiting "test" group for now)
# X_train, X_test, y_train, y_test = train_test_split(nifti_images, labels, test_size=0.5, random_state=43)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

# print(X_train.shape)

# #### model
# model = models.Sequential()
# # First Convolutional Layer 
# model.add(Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)))
# # model.add(layers.Conv3D(8, (3, 3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],1))) ## 1 in 4 dim represens channel dim, is only one b/c it's a binary mask.
# model.add(layers.Conv3D(8, (3, 3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling3D((2, 2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(8, activation='relu')) #changed from 64 down to 8, 64 was overkill
# model.add(layers.Dense(2, activation='sigmoid'))  # Assuming 2 classes, modify as needed
# # model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes, modify as needed

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=5, batch_size=8, validation_data=(X_test, y_test))

# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc}')

# #### model save
# id_bub = random.randint(1000,9999)
# print(id_bub)
# model.save("/rd1_model_rev1_run"+str(id_bub)+".keras") #### <----- I used this to save w/ test_size=0.45

# #### optional stuff



# # Second Convolutional Layer 
# # model.add(layers.Conv3D(8, (3, 3, 3), activation='relu'))
# # model.add(layers.BatchNormalization())
# # model.add(layers.MaxPooling3D((2, 2, 2)))
# # # Third Convolutional Layer 
# # model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
# # model.add(layers.BatchNormalization())
# # model.add(layers.MaxPooling3D((2, 2, 2)))
# # Flatten and Dense layers


