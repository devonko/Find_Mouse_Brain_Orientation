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

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback


state_1 = 0 ## <-- this is some custom-built stuff to terminate the tensor flow model upon reaching a designated accuracy level
class TerminateOnBaseline(Callback):
    def __init__(self, baseline):
        super(TerminateOnBaseline, self).__init__()
        self.baseline = baseline
    def on_epoch_end(self, epoch, logs=None):
        global state_1
        if logs.get('val_accuracy') >= self.baseline:
            if logs.get('accuracy') >= self.baseline:
                state_1 +=1
                print(f"\nReached 1st {self.baseline*100}% train & validation accuracy, working for reducancy...")
                if state_1 == 2:
                    print(f"\nReached 2nd {self.baseline*100}% train & validation accuracy. Stopping training...")
                    print(state_1)
                    self.model.stop_training = True
                    return state_1
        

print("Loading master spreadsheet instructions...")
path1 = ["parent_path"]
df = pd.read_csv(str(path1[0])+"/modeling/master_spreadsheet2.csv")

filtered_df = df[df['test_group_classification'] == 0]
nifti_files = filtered_df['file_name'].tolist()
labels =  filtered_df['orientation_category'].tolist()
unique_count = filtered_df['orientation_category'].nunique()
print(unique_count)

print("Loading NIFTI images...")
nifti_images=[]
for i in range(len(nifti_files)):
    # img = nib.load(str(path1[0])+"/github/test_masks/"+str(nifti_files[i]))
    img = nib.load(str(path1[0])+"/modeling/resampled_masks/"+str(nifti_files[i]))
    nifti_images.append(img.get_fdata())  # Convert NIfTI image to numpy array

print("Separating into train and validate groups...")
#### separate into train/validate (omiting "test" group for now)
X_train, X_test, y_train, y_test = train_test_split(nifti_images, labels, test_size=0.5, random_state=12)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Size of X_train:")
print(X_train.shape)
print(y_train.shape)

print("Size of X_test:")
print(X_test.shape)
print(y_test.shape)

print("modeling...")

terminate_on_baseline = TerminateOnBaseline(baseline=1.00) ### <--- accuracy threshold
# #### model
model = models.Sequential()
# First Convolutional Layer 
model.add(Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)))

model.add(layers.Conv3D(32, (3, 3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.2)) ## <- trying to force a more generalizeable model, might be overfitting with 0.05 levels

model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.2)) ## <- trying to force a more generalizeable model, might be overfitting with 0.05 levels

model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Dropout(0.2)) ## <- trying to force a more generalizeable model, might be overfitting with 0.05 levels

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))) 
model.add(layers.Dropout(0.3)) ## <- trying to force a more generalizeable model, might be overfitting with 0.1 level

model.add(layers.Dense(24, activation='softmax'))  # Assuming 24 classes, modify as needed

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded, sparse_categorical_crossentropy, binary_crossentropy
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=75, batch_size=4, validation_data=(X_test, y_test),callbacks=[terminate_on_baseline])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# #### model save
if state_1 == 2:
    id_bub = random.randint(1000,9999)
    print(id_bub)
    model.save(str(path1[0])+"/s2_tf_model_rev1_run"+str(id_bub)+".keras") 
