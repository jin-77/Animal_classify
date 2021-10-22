import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
from keras.utils.np_utils import to_categorical
from PIL import Image
import sys, os, time
import streamlit as st
st.title('動物分類アプリ')
st.header('サル・ネコ・カラスを判定しよう！')

classes = ['monkey','cat','crow']
num_classes = len(classes)
image_size=150


uploaded_file = st.file_uploader('Choose an image of monkeys, boars or crows...', type = ['jpg','png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)
    img_path = f'img/{uploaded_file.name}'
    img.save(img_path)
    model = load_model('./animal_cnn_augnew1.h5')
    image = Image.open(img_path)
    image = image.convert('RGB')
    image = image.resize((image_size,image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    
    st.write()
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range (100):
        latest_iteration.text(f'Now loading.... {i+1}')
        bar.progress(i + 1)
        time.sleep(0.01)
    'Done!!'

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted]*100)
    st.markdown( '動物の種類：' + classes[predicted] + '，確率：' +  str(percentage) + '%')
            
