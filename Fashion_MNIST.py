# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Array to store the items of different images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Exploring images - training images
#Result: (60000,28,28)
#Explanation: 60000 refers to the number of training images, with each images
#being a 28x28 image
print(train_images.shape)

#Training labels
#Result: 60000
#Explanation: There are 60000 labels in the training images
print(len(train_labels))

#Training labels pt.2
#Result: [9,0,0,...3,0,5]
#Explanation: Each label is labelled by an integer between 0-9
print(train_labels)

#Test_images
print(test_images.shape)
print(len(test_images))

#Preprocessing data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#Scale the images so that they range from 0-1
train_images = train_images/255.0
test_images = test_images/255.0

#Create a grid that displays 25 images from the training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#Build the neural network by starting with layers. Layers extract representations
#from the data given to them and hopefully there's some meaning

#Flatten changes the image that was a 28x28 2d array into a 1d 28*28 = 784 pixel
#array

#The two Dense layers have to do with the number of nodes. The second Dense layers
#indicates the score of the probability that the image will be of one of ten classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

#Training the model: First there are settings that need to be made
#Optimizer: How the model is updated based on data it sees and the loss function
#Loss: The loss function is minimized so that the model can be "steered" in the
#right direction
#Metrics: Used to monitor the training and testing of the model
model.compile(optimizer=tf.train.AdamOptimizer(),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

#The model can be trained using model.fit. The model will make predictions and
#associate labels with images
model.fit(train_images,train_labels,epochs=5)

#Evalute the accuracy of the model by using the test images
test_loss,test_accuracy = model.evaluate(test_images,test_labels)
print('Test accuracy: ',test_accuracy)

#The preceding was an example of overfitting. Where the model performs worse
#with the test images than the training images

#Now the model can try making predictions
predictions = model.predict(test_images)
print(predictions[0])

#We can find the label where the model is the most confident about for being the
#correct one
print(np.argmax(predictions[0]))

#We can also check if its prediction was correct by calling the image in the Array
#based on the prediction array index of the model
print(test_labels[0])

#Make a graph based on the predictions of the confidence for each label
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#Example with the 0th image in the Array
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

#Example with the 12th image in the array
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

#Image predictions
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

#Make predictions
predictions_single = model.predict(img)

print(predictions_single)

#Create the graph to display the predictions
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

plt.show()
