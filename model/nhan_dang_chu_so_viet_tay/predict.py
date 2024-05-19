import tensorflow as tf
from tensorflow.keras import models, optimizers, datasets
import cv2
import numpy as np

(_, _), (X_test, _) = datasets.mnist.load_data()

X_test = X_test.reshape((10000, 28, 28, 1))

sample = X_test[500,:,:,0]
cv2.imshow('Digit', sample)
sample = sample / 255.0
sample = sample.astype('float32')
#Them truc phia truoc
sample = np.expand_dims(sample, axis=0)
#Them truc phia sau
sample = np.expand_dims(sample, axis=3)
print('Shape =', sample.shape)

OPTIMIZER = tf.keras.optimizers.Adam()

model_config = 'digit_config.json'
model_weight = 'digit_weight.h5'

model = models.model_from_json(open(model_config).read())
model.load_weights(model_weight)

model.compile(loss="categorical_crossentropy", 
              optimizer=OPTIMIZER,
              metrics=["accuracy"])
#model.summary()
ket_qua = model.predict(sample)
print(ket_qua)
chu_so = np.argmax(ket_qua[0])
print(chu_so)
cv2.waitKey()