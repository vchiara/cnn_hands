import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D

import tensorflow_hub as hub

#load dataset
df = pd.read_csv('HandInfo.csv')

!unzip 'Hands.zip'

#explore dataset
!ls Hands

df.columns

df['skinColor'].value_counts().plot.bar()

#distribution graphs
df['gender'].value_counts().plot.bar()

df['age'].hist(bins=50)

df['aspectOfHand'].value_counts().plot.bar()

#select and encode vars
encoder = LabelEncoder()
encoder.fit(df['aspectOfHand'])
df['y'] = encoder.transform(df['aspectOfHand'])
df

x = df['imageName']
y = df['y']

dataset = Dataset.from_tensor_slices((x, y))

#load images
def process_image(image_path, label):
  #load file
  image_raw = tf.io.read_file('Hands/' + image_path)
  image_decoded = tf.image.decode_jpeg(image_raw)

  #resize and scaling
  image_resized = tf.image.resize(image_decoded, [224, 224])
  image_scaled = image_resized / 255.0

  #return image and label
  return image_scaled, label

dataset = dataset.map(process_image)

for image, label in dataset.take(10):
  print(image, label)

for image, label in dataset.take(20): #plot images
  #print(image,label)
  plt.imshow(image)
  plt.title(label.numpy())
  plt.show()

#split into train and test sets
train_size = int(len(df) * 0.8)
train_size

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

train_dataset.cardinality().numpy(), test_dataset.cardinality().numpy()

#batch
train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

train_dataset.cardinality().numpy(),test_dataset.cardinality().numpy()

train_dataset



##cnn model
model = Sequential([
  Conv2D(16, 3, activation = 'relu', padding = 'same', input_shape = (256, 256, 3)),
  MaxPool2D(),

  Conv2D(16, 3, activation = 'relu', padding = 'same'),
  MaxPool2D(),

  Conv2D(16, 3, activation = 'relu', padding = 'same'),
  MaxPool2D(),

  Conv2D(16, 3, activation = 'relu', padding = 'same'),
  MaxPool2D(),

  Conv2D(16, 3, activation = 'relu', padding = 'same'),
  MaxPool2D(),

  Flatten(),
  Dropout(0.2),
  Dense(4, activation = 'softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()
#Total params: 13,828
#Trainable params: 13,828
#Non-trainable params: 0

model.fit(train_dataset, epochs = 10)
#Epoch 10/10 554/554 [==============================] - 72s 130ms/step - loss: 0.0778 - accuracy: 0.9757

model.evaluate(test_dataset)
#[1.0396385192871094, 0.8921480178833008]

#load new image and predict label
new_image, fake_label = process_image('Hand_0000095.jpg', 0)
plt.imshow(new_image)

new_batch = tf.expand_dims(new_image, axis = 0)

p = model.predict(new_batch)
p

pred = np.argmax(p[0])

encoder.inverse_transform([pred])
#palmar right



##model with transfer learning

#load pretrained model
pretrained_model_url = 'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5'

#build cnn
pretrained_model = Sequential([
  hub.KerasLayer(pretrained_model_url,
                 trainable = False,
                 input_shape = [224, 224, 3]),
  Dropout(0.4),
  Dense(4, activation = 'softmax')
])

pretrained_model.summary()

pretrained_model.compile(
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#train model
pretrained_model.fit(train_dataset, epochs = 10)
#Epoch 10/10 554/554 [==============================] - 62s 112ms/step - loss: 0.1772 - accuracy: 0.9354

#test model
pretrained_model.evaluate(test_dataset)
#[0.4444352984428406, 0.833032488822937]

#fine tuning model
pretrained_model.layers[0].trainable = True

pretrained_model.fit(train_dataset, epochs = 10)

pretrained_model.evaluate(test_dataset)
#[0.4672292470932007, 0.8524368405342102]

#predict label
pretrained_p = pretrained_model.predict(new_batch)
pretrained_p

pretrained_pred = np.argmax(pretrained_p[0])

encoder.inverse_transform([pretrained_pred])
#palmar right
