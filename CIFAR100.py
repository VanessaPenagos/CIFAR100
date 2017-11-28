from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import cifar100
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import GridSearchCV

def create_model(learning_rate=1e-3, dropout_rate=0.2, units=8):
    model = Sequential()
    model.add(Conv2D(filters=units, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=2 * units, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=4 * units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=8 * units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=100, activation='softmax'))
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

batch_size = 32
training_size = X_train.shape[0]

model = KerasClassifier(
    build_fn=create_model,
    epochs=5,
    steps_per_epoch=training_size//batch_size,
    verbose=0)

param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'units': [4, 8, 16, 32]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid.fit(X_train, y_train)