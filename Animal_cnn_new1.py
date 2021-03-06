import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
#from keras.layers import np_utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping




classes = ['monkey','cat','crow']
num_classes = len(classes)
image_size=150

#メインの関数を定義する
def main():
    X_train,X_test,y_train,y_test= np.load('./animal_augnew1.npy',allow_pickle=True)
    X_train = X_train.astype('float')/256
    X_test = X_test.astype('float')/256
    y_train = to_categorical(y_train,num_classes)
    y_test = to_categorical(y_test,num_classes)

    model = model_train(X_train,y_train)
    model_eval(model,X_test,y_test)

def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, callbacks=[EarlyStopping(monitor='loss',
                            min_delta=0.0,patience=1)], batch_size=32, epochs=50)

    # モデルの保存
    model.save('./animal_cnn_augnew1.h5')

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()
