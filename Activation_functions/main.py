from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam
from imutils import paths
from time import time
import numpy as np
import cv2
import os


# Initialize variables
input_shape = (224, 224, 3)
num_classes = 5
epochs = 5

dataset_training = "Imagenet/"
""" path_model = "/content/drive/My Drive/Dataset/Keras/model_Keras/model.model"
label_bin = "/content/drive/My Drive/Dataset/Keras/model_Keras/label.pickle" 
results = "/content/drive/My Drive/Dataset/Keras/results_Keras/"
path_plots = "/content/drive/My Drive/Dataset/Keras/plots_Keras/" """

# Methods

def activation_tanh(x):
    return x*K.tanh(x)

# Initialize the set of labels from the Imagenet dataset
LABELS = set(["acorn", "goldfish", "lemon", "police_van", "seashore"])

print("Loading images")
imagePaths = list(paths.list_images(dataset_training))
data = []
labels = []


# Loop over the image paths
for imagePath in imagePaths:
    # Extract the class label from the filename
    path_image = imagePath.split(os.path.sep)
    label = path_image[-2]

    # If the label of the current image is not part of of the labels are interested in, then ignore the image
    if label not in LABELS:
        continue

    # Load the image, convert it to RGB channel ordering, and resize it to be a fixed 224x224 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Assign the training and the test data
# The aim of the random_state param is to get always the same result -
# https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# Building a CNN
""" model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  """

# Building a VGG16
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=num_classes, activation="softmax"))

# Building a customized CNN
""" model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape))
model.add(Activation(activation_tanh))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation(activation_tanh))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation(activation_tanh))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation(activation_tanh))
model.add(Dense(num_classes, activation='softmax')) """

# Compiling the model
print("Compiling model")

# Using Adam optimizer
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.009, epsilon=0.1)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

# Training the model
print("Training model")
start_time = time()
""" H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=epochs) """

history = model.fit(trainX, trainY, batch_size=32, epochs=epochs, validation_data=(testX, testY))
end_time = time()
# Calculating training time
total_time = end_time - start_time

# Evaluating the CNN
print("Evaluating network")
predictions = model.predict(testX, batch_size=32)
# Saving the classification report into a csv file with pandas
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# classification_report_csv(report, filename_class, num_classes)


print("END")