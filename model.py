import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Cropping2D, Lambda, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess_path(path, ratio=1):
    '''
    create lists of file path, steering angle and image mirroring flag
    '''
    col_names = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
    df = pd.read_csv(path + 'driving_log.csv', header=None, names=col_names)

    img_path_list = []
    measurements = []
    flip = []

    for shot_idx in range(int(len(df)*ratio)):
        for loc, correction in zip(['Center Image', 'Left Image', 'Right Image'],
                                   [0.0, 0.2, -0.2]):
            img_path_list.append(path + 'IMG/' + df[loc][shot_idx].split('/')[-1])
            measurements.append(float(df['Steering Angle'][shot_idx]) + correction)
            flip.append(False)

            img_path_list.append(path + 'IMG/' + df[loc][shot_idx].split('/')[-1])
            measurements.append(float(df['Steering Angle'][shot_idx]) + correction)
            flip.append(True)

    return img_path_list, measurements, flip

def generator(df, batch_size=32):
    '''
    Data generator, generates batches of size batch_size indefinitely
    '''
    num_samples = len(df)
    while 1: # Loop forever so the generator never terminates
        df = shuffle(df)
        for offset in range(0, num_samples, batch_size):
            batch_samples = df.iloc[offset:offset+batch_size,:]

            images = []
            angles = []
            for idx in range(len(batch_samples)):
                img = cv2.imread(batch_samples['Path'].iloc[idx])
                angle = float(batch_samples['Steering'].iloc[idx])

                if batch_samples['Flip'].iloc[idx]:
                    images.append(np.fliplr(img))
                    angles.append(-angle)
                else:
                    images.append(img)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# images files of driving forward (counter-clockwisely) over track 1 for three laps
path1, measure1, flip1 = preprocess_path('driving_data/track1_fwd_v2/', ratio=1)
# images files of driving in reverse direction (clockwisely) over track 1 for three laps
path2, measure2, flip2 = preprocess_path('driving_data/track1_rev_v2/', ratio=1)
# images files of driving in forward and reverse making turns (only making turns)
# ratio=0.5 implies only use the image files of driving forward
path4, measure4, flip4 = preprocess_path('driving_data/track1_smooth_turn_v2/', ratio=0.5)
#path3, measure3, flip3 = preprocess_path('driving_data/track1_recovery/', ratio=0.5)
# image files in the sample training data on the class website
path5, measure5, flip5 = preprocess_path('driving_data/data/')

# create dataframe
df = pd.DataFrame({'Path': pd.Series(path1 + path2 + path4 + path5),
                   'Steering': pd.Series(measure1 + measure2 + measure4 + measure5),
                   'Flip': pd.Series(flip1 + flip2 + flip4 + flip5)})
# split the dataframe into training and validation frames
train_df, valid_df = train_test_split(df, test_size=0.2)

# create the data generators
train_generator = generator(train_df, batch_size=32)
valid_generator = generator(valid_df, batch_size=32)

# create the model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(BatchNormalization(axis=3))
model.add(Activation('elu'))
#model.add(Dropout(p=0.2))
model.add(Conv2D(32, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(BatchNormalization(axis=3))
model.add(Activation('elu'))
#model.add(Dropout(p=0.2))
model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
model.add(BatchNormalization(axis=3))
model.add(Activation('elu'))
#model.add(Dropout(p=0.2))
model.add(Conv2D(64, 3, 3, border_mode='valid'))
model.add(BatchNormalization(axis=3))
model.add(Activation('elu'))
#model.add(Dropout(p=0.2))
model.add(Conv2D(64, 3, 3, border_mode='valid'))
model.add(BatchNormalization(axis=3))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dropout(p=0.5))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(p=0.2))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(p=0.2))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('elu'))
#model.add(Dropout(p=0.1))
model.add(Dense(1))

# specify the optimizer and compile the model
adam = Adam(lr=1e-3)
model.compile(optimizer=adam, loss='mse')

# fit the model for five epochs
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_df),
                                     validation_data=valid_generator,
                                     nb_val_samples=len(valid_df), nb_epoch=5)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# save the model
model.save('model.h5')