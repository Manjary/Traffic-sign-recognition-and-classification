import tensorflow as tf

import cv2
from skimage import data
from skimage import transform
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import random
import matplotlib.pyplot as plt


# Load pickled data
import pickle

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as file:
    train = pickle.load(file)
with open(testing_file, mode='rb') as file:
    test = pickle.load(file)
    



X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print('Train size -',len(train['features']))
print('Test size -',len(test['features']))


# Defining function to transform images and generate more images

def transform_image(img):
    rows,cols,ch = img.shape
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pts2 = np.float32([[0,10],[20,5],[10,25]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


# Plotting histogram showing count of images in each class


n_classes = len(np.bincount(y_train))
plt.hist(y_train, bins=n_classes, color='red')

# Adding more training images by transforming training set

class_mean = int(np.mean(np.bincount(y_train)))
num_of_classes = len(np.bincount(y_train))
print(class_mean,num_of_classes)
count_in_class = np.bincount(y_train)
for i in range(num_of_classes):
    if count_in_class[i] < class_mean:
        new_image_needed = class_mean - count_in_class[i]
        index = np.where(y_train == i)
        additional_X = []
        additional_y = []
        for count in range(new_image_needed):
            additional_X.append(transform_image(X_train[index][random.randint(0,count_in_class[i] - 1)]))
            additional_y.append(i)
        print('Additional images',len(additional_X),'for class',i )

        X_train = np.append(X_train, np.array(additional_X), axis=0)
        y_train = np.append(y_train, np.array(additional_y), axis=0)



#  Number of training examples
n_train = len(X_train)

#  Number of testing examples.
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(np.bincount(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

plt.hist(y_train, bins = n_classes,color = 'red')


# Gray scaling images

from numpy import newaxis
import cv2


def do_grayscale(img_arr):
    new_img_arr = []
    for img in img_arr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_img_arr.append(gray)
        
    return np.array(new_img_arr)


X_train = do_grayscale(X_train)
X_train = X_train[..., newaxis]

X_test = do_grayscale(X_test)
X_test = X_test[..., newaxis]

print(X_train.shape,X_test.shape)


##  Design and Test a Model Architecture - Design and implement a deep learning model that learns to recognize traffic signs.

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


# Normalize the data features to the variable X_normalized
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


X_train = normalize_grayscale(X_train)
X_test = normalize_grayscale(X_test)
print(X_train.shape)



from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size = .2, random_state = 0)
print('Train-',len(X_train),'Validation-',len(X_validation))

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))


import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128


# Main LeNet function definition

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.05
    keep_prob = 0.6
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # dropout to the neural network
    # conv1 = tf.nn.dropout(conv1, keep_prob)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    #  Activation.
    conv2 = tf.nn.relu(conv2)

    # dropout to the neural network
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    
    #  Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    #  Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    #  Activation.
    fc1    = tf.nn.relu(fc1)

    #  Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    #  Activation.
    fc2    = tf.nn.relu(fc2)

    #  Layer 5: Fully Connected. Input = 84. Output = 64.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 64), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(64))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b
    
    #  Activation.
    fc3    = tf.nn.relu(fc3)
    
    #  Layer 5: Fully Connected. Input = 64. Output = 43.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(64, 43), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc3, fc4_W) + fc4_b
    
    return logits

#Features and Labels
#x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)



### Training model here.


rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num


# Train the model
# Run the training data through the training pipeline to train the model.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")
    

# Running on test dataset
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    accu = evaluate(X_test, y_test)
    print("Test Accuracy % = {:.3f}".format(accu))
