import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import MaxPooling1D, LeakyReLU, Conv2D, Conv1D, Input, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import pickle

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def train(symbols, dates, test_size=0.2, patience=10, undersample=True, restore=True):

    for symbol in symbols:

        for d in dates:

            path = f'../data/{symbol.upper()}/{d}'
            X, y = pickle.load(open(f'{path}/supervised.pickle', 'rb'))

            print(np.unique(y, axis=0, return_counts=True))

            if undersample:
                values = np.unique(y, axis=0, return_counts=True)
                lowest_index = values[1].argmin()
                lowest_value = values[1].min()

                for i in range(3):
                    if i == lowest_index:
                        continue
                    mask = np.where(np.all(y == values[0][i], axis=1))[0]
                    choice = np.random.choice(mask, values[1][i] - lowest_value, replace=False)
                    X = np.delete(X, choice, axis=0)
                    y = np.delete(y, choice, axis=0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Convolutional Neural Network
            input_image = Input(shape=X.shape[1:])
            x = Conv2D(16, (4, X.shape[2]), input_shape=X.shape[1:])(input_image)
            x = LeakyReLU()(x)
            x = tf.squeeze(x, 2)
            x = Conv1D(16, 4)(x)
            x = LeakyReLU()(x)
            x = MaxPooling1D(2)(x)
            x = LeakyReLU()(x)
            x = Conv1D(32, 3)(x)
            x = LeakyReLU()(x)
            x = Conv1D(32, 3)(x)
            x = LeakyReLU()(x)
            x = MaxPooling1D(2)(x)
            x = LeakyReLU()(x)
            x = Flatten()(x)
            x = Dense(32)(x)
            x = LeakyReLU()(x)
            x = Dropout(0.1)(x)
            out = Dense(3, activation='softmax')(x)
            model = Model(inputs=input_image, outputs=out)

            # Fit
            sgd = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False, name="SGD")
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=restore)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=128, callbacks=[es])

            # Evaluate
            print(model.evaluate(X_test, y_test))
            print(confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(model.predict(X_test), axis=1)))

            # Save
            model.save(f"{path}/model.h5")


if __name__ == "__main__":
    train(['BTCUSDT'], ['2021_03_12'], test_size=0.1, patience=15, restore=True, undersample=False)
