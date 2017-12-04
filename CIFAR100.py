from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import cifar100
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import GridSearchCV

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def create_model(learning_rate=1e-3, dropout_rate=0.1, units=8):
    model = Sequential()
    model.add(Conv2D(filters=units, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=units, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=2 * units, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=2 * units, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=4 * units, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=4 * units, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(units=10 * units))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=100, activation='softmax'))
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

epochs = 1 #300
batch_size = 5 #32
training_size = X_train.shape[0]

param_grid = {
    'learning_rate': [1e-3, 1e-2],
    'dropout_rate': [0.2, 0.4, 0.8],
    'units': [4, 8, 16]
}

print(param_grid)

model = KerasClassifier(
    build_fn=create_model,
    epochs=epochs,
    steps_per_epoch= 2, #training_size // batch_size,
    verbose=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv=2)
grid.fit(X_train[:5], y_train[:5])

print("Best Score: ",grid.best_score_, " Best Params: ", grid.best_params_)

means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']

for mean, param in zip(means, params):
    print("%f with: %r" %(mean, param))

# PRUEBA SIN GRIDSEARCH
# model = create_model()
# checkpoint = ModelCheckpoint('checkpoints/epoch_{epoch:02d}-valacc_{val_acc:.2f}.hdf5', period=100)
# model.fit(X_train, y_train,
#           validation_split=0.2,
#           epochs=epochs, batch_size=batch_size,
#           callbacks=[checkpoint])
# model.save('model.hdf5')
