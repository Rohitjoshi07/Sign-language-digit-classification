import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,  Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import  Adam



def get_data(choice):
    Y = np.load("correct_Data/Y_new.npy")
    X = np.load("correct_Data/X_new.npy")

    X = X.reshape(-1, 64, 64, 1)
    print('X shape : {}  Y shape: {}'.format(X.shape, Y.shape))

    ### Randomize dataset
    shuffle_index = np.random.permutation(2062)
    X, Y = X[shuffle_index], Y[shuffle_index]
    trainx, valx, trainy, valy = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=8)
    if choice == 'whole':
        return X, Y
    else:
        return trainx, valx, trainy, valy


def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return


def model_cnn():
    num_class = 10
    image_size = 64
    channel_size = 1

    model = Sequential()
    # Layer_1:- conv2d--->pool---->batchnorm------>dropout
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), input_shape=(image_size, image_size, 1),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer2:-conv2d--->pool---->batchnorm------>dropout
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     padding='valid'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer3:- conv2d--->pool---->batchnorm------>dropout
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer3:- conv2d--->pool---->batchnorm------>dropout
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 5:- flatten and dense

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    # Layer 6
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())

    # layer 7
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())

    # layer 8
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())

    # layer 9
    model.add(Dense(num_class, activation='softmax'))

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    return model


def show_model_history(modelHistory, model_name):
    history = pd.DataFrame()
    history["Train Loss"] = modelHistory.history['loss']
    history["Validation Loss"] = modelHistory.history['val_loss']
    history["Train Accuracy"] = modelHistory.history['accuracy']
    history["Validation Accuracy"] = modelHistory.history['val_accuracy']

    fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    axarr[0].set_title("Loss curve of Train and Validation data")
    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])

    axarr[1].set_title("Accuracy curve of Train and Validation Data")
    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1])

    plt.suptitle("Model {} Loss and Accuracy in Train and Validation Data".format(model_name))
    plt.show()
