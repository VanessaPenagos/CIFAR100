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

epochs = 1000
batch_size = 32

model = create_model()
checkpoint = ModelCheckpoint('checkpoints/epoch_{epoch:02d}-valacc_{val_acc:.2f}.hdf5', period=100)
model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=epochs, batch_size=batch_size,
          callbacks=[checkpoint])
model.save('model.hdf5')

#model = KerasClassifier(
#    build_fn=create_model,
#    epochs=5,
#    steps_per_epoch=training_size//batch_size,
#    verbose=0)

#param_grid = {
#    'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
#    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
#    'units': [4, 8, 16, 32]
#}
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid.fit(X_train, y_train)
