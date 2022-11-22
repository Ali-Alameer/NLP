globals().clear()  # clear all variables

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# set the percentages of training, validation and testing dataset
test_percentage = 0.1
validation_percentage = 0.1

# read data from csv file instead (dataset available in github)
data = pd.read_csv('directory/of/your/data/spam_text/Data.csv')
X = data.Message.values
y = data.Category.values

# convert the labels into numbers
y[y == 'spam'] = 0
y[y == 'ham'] = 1
y = np.asarray(y).astype('float32')

# Partition the data into training and testing
test_examples = np.asarray(X[:round(test_percentage * len(X))])
train_examples = np.asarray(X[round(test_percentage * len(X)):])

test_labels = np.asarray(y[:round(test_percentage * len(X))])
train_labels = np.asarray(y[round(test_percentage * len(X)):])

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))

# show subsample ofo the training examples and training labels
train_examples[:10]
train_labels[:10]

# Build the model and show its layers; model has two fully connected layers with hidden units of 16 and 1, respectively
model = "https://tfhub.dev/google/nnlm-en-dim50/2"
# model = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
# model = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"

hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples[:3])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))  # because it's binary classification
model.summary()

# the loss function and metric are compatible with binary classification scenarios
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

# extracting validation examples from the training data
x_val = train_examples[:round(validation_percentage * len(train_examples))]
partial_x_train = train_examples[round(validation_percentage * len(train_examples)):]

y_val = train_labels[:round(validation_percentage * len(train_examples))]
partial_y_train = train_labels[round(validation_percentage * len(train_examples)):]

# training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# see model training history
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



results = model.evaluate(test_examples, test_labels)  # this return loss value and accuracy
print(results)

# another method to evaluate performance
predictions = model.predict(test_examples)
predictions[predictions >= 0] = 1
predictions[predictions < 0] = 0
confusionMatrix = confusion_matrix(test_labels, predictions, normalize='pred')
acc = metrics.accuracy_score(test_labels, predictions)
print(classification_report(test_labels, predictions))