import tensorflow as tf
from tensorflow.keras import models
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape, Y_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = define_model()
history = model.fit(X_train, Y_train, epochs=10, batch_size=64,
                    validation_data=(X_test, Y_test), verbose=1)


model.save('mnist.h5')
print("Saving the model as mnist.h5")

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
