# ALL-IN-ONE
PRACTICAL NO: 01
Aim: Performing matrix multiplication and finding eigen vectors and eigen values using TensorFlow.

import tensorflow as tf
print("Matrix Multiplication Demo")

x=tf.constant([1,2,3,4,5,6],shape=[2,3])
print(x)

y=tf.constant([7,8,9,10,11,12],shape=[3,2])
print(y)

z=tf.matmul(x,y)
print("Product:" ,z)

e_matrix_A=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,name="matrixA")
print("Matrix A:\n{}\n\n", format(e_matrix_A))

eigen_values_A,eigen_vectors_A=tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors:\n", eigen_vectors_A)
print("Eigen Values:\n", eigen_values_A)

PRACTICAL NO: 02
Aim: Solving XOR problem using deep feed forward network.
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Initialize the model
model = Sequential()

# Add the first Dense layer
model.add(Dense(units=2, activation="relu", input_dim=2))

# Add the second Dense layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Print the model's initial weights (before training)
print("Initial weights:", model.get_weights())

# Define the training data
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.])

# Train the model
model.fit(X, Y, epochs=1000, batch_size=4)

# Print the model's final weights (after training)
print("Final weights:", model.get_weights())

# Make predictions using the trained model
predictions = model.predict(X, batch_size=4)
print("Predictions:", predictions)

# PRACTICAL NO: 03
# Aim: Implementing deep neural network for performing classification task.
# Code:
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = loadtxt('C:/Users/admin/Documents/block/pythonProject1/pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Print the input data
print(X)
print(Y)

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train the model
model.fit(X, Y, epochs=150, batch_size=10)

accuracy = model.evaluate(X, Y)
print("Accuracy of model is", (accuracy * 100))

# Predict the output for the training data
prediction = model.predict(X, batch_size=4)

# Print the predictions
print(prediction)
exec("for i in range(5):print(X[i].tolist(),prediction[i], Y[i])")

# PRACTICAL NO: 04
# A) Aim: Using deep feed forward network with two hidden layers for performing classification and predicting the class .
#
# Code:

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Generate dataset
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Scale the dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500)

# Generate new data for prediction
X_new, Y_real = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
X_new = scaler.transform(X_new)

# Predict the class for new data
Y_new = np.argmax(model.predict(X_new), axis=1)
for i in range(len(X_new)):
	print("X=%s, Predicted=%s, Desired=%s" % (X_new[i], Y_new[i], Y_real[i]))
 
# B) Aim: Using a deep field forward network with two hidden layers for
# performing classification and predicting the probability of class
# Code:

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Generate dataset
X, Y = make_blobs(n_samples=100, centers=2, n_features=2,
                  random_state=1)

# Scale the dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the appropriate loss function for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500)

# Generate new data for prediction
X_new, Y_real = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
X_new = scaler.transform(X_new)

# Predict the class and probability for new data
Y_class = (model.predict(X_new) > 0.5).astype("int32")
Y_prob = model.predict(X_new)

for i in range(len(X_new)):
	print("X=%s, Predicted_probability=%s, Predicted_class=%s" %(X_new[i], Y_prob[i], Y_class[i]))

# C) Aim: Using a deep field forward network with two hidden layers for
# performing linear regression and predicting values.
# Code:

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
X,Y=make_regression(n_samples=100,n_features=2,noise=0.1,random_state=1)
scalarX,scalarY=MinMaxScaler(),MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape(100,1))
X=scalarX.transform(X)
Y=scalarY.transform(Y.reshape(100,1))
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')
model.fit(X,Y,epochs=1000,verbose=0)
Xnew,a=make_regression(n_samples=3,n_features=2,noise=0.1,random_state=1)
Xnew=scalarX.transform(Xnew)
Ynew=model.predict(Xnew)
for i in range(len(Xnew)):
 print("X=%s,Predicted=%s"%(Xnew[i],Ynew[i]))

 # A) Aim: Evaluating feed forward deep network for regression using KFold
# cross validation.
# Code:

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate sample data
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1)
# Define KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize list to store evaluation metrics
eval_metrics = []
# Iterate through each fold
for train_index, test_index in kfold.split(X):
	# Split data into training and testing sets
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	# Define and compile model
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=10))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])
	# Fit model to training data
	model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
	# Evaluate model on testing data
	eval_metrics.append(model.evaluate(X_test, y_test))

# Print average evaluation metrics across all folds
print("Average evaluation metrics:")
print("Loss:", np.mean([m[0] for m in eval_metrics]))
print("MAE:", np.mean([m[1] for m in eval_metrics]))

# B) Aim: Evaluating feed forward deep network for multiclass Classification
# using KFold cross-validation.
# Code:

import pandas
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# Import to_categorical directly from tensorflow.keras.utils
from tensorflow.keras.utils import to_categorical

# loading dataset
# Tell pandas to treat the first row as a header
df = pandas.read_csv('C:/Users/admin/Documents/block/pythonProject1/flowers.csv',header=0)
print(df)
# splitting dataset into input and output variables
# Adjust column indexing to start from 0
X = df.iloc[:, 0:4].astype(float)
y = df.iloc[:, 4]
# print(X)
# print(y)
# encoding string output into numeric output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print(encoded_y)
# Use to_categorical from tensorflow.keras.utils
dummy_Y = to_categorical(encoded_y)
print(dummy_Y)


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = baseline_model()
estimator.fit(X, dummy_Y, epochs=100, shuffle=True)
action = estimator.predict(X)
for i in range(25):
	print(dummy_Y[i])
	print('^^^^^^^^^^^^^^^^^^^^^^')
for i in range(25):
	print(action[i])

# PRACTICAL NO: 06
# A) Aim: implementing regularization to avoid overfitting in binary classification.
# Code:


from matplotlib import pyplot
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
X,Y=make_moons(n_samples=100,noise=0.2,random_state=1)
n_train=30
trainX,testX=X[:n_train,:],X[n_train:]
trainY,testY=Y[:n_train],Y[n_train:]
#print(trainX)
#print(trainY)
#print(testX)
#print(testY)
model=Sequential()
model.add(Dense(500,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=4000)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test')
pyplot.legend()
pyplot.show()

# B) Aim: Implementing L2 regularization
# Code:

#Implementing L2 regularization
from matplotlib import pyplot
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:]
trainY, testY = Y[:n_train], Y[n_train:]
#print(trainX)
#print(trainY)
#print(testX)
#print(testY)
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=4000)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# C) Aim: Replacing L2 Regularizer with L1 Regularizer
# Code:

#Replacing L2 regularizer with L1 regularizer
from matplotlib import pyplot
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1_l2
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:]
trainY, testY = Y[:n_train], Y[n_train:]
#print(trainX)
#print(trainY)
#print(testX)
#print(testY)
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer = l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=4000)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# PRACTICAL NO: 07
# Aim: Demonstrate recurrent neural network that learns to perform sequence
# analysis for stock price.
# Code:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
dataset_train=pd.read_csv('C:/Users/admin/Documents/block/pythonProject1/Google_Stock_Price_Train.csv')
#print(dataset_train)
training_set=dataset_train.iloc[:,1:2].values
#print(training_set)
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#print(training_set_scaled)
X_train=[]
Y_train=[]
for i in range(60,1258):
 X_train.append(training_set_scaled[i-60:i,0])
 Y_train.append(training_set_scaled[i,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)
print(X_train)
print('*********************************************')
print(Y_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
print('**********************************************')
print(X_train)
regressor=Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,Y_train,epochs=100,batch_size=32)
dataset_test=pd.read_csv('C:/Users/admin/Documents/block/pythonProject1/Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(60,80):
 X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color='red', label='real google stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()

# PRACTICAL NO: 08
# Aim: Performing encoding and decoding of images using deep autoencoder.
# Code:


import keras
#from tensorflow.keras import layers
from keras.layers import Dense, Input
from keras import layers
from keras.datasets import mnist
import numpy as np
encoding_dim=32
#this is our input image
input_img=keras.Input(shape=(784,))
#"encoded" is the encoded representation of the input
encoded=layers.Dense(encoding_dim, activation='relu')(input_img)
#"decoded" is the lossy reconstruction of the input
decoded=layers.Dense(784, activation='sigmoid')(encoded)
#creating autoencoder model
autoencoder=keras.Model(input_img,decoded)
#create the encoder model
encoder=keras.Model(input_img,encoded)
encoded_input=keras.Input(shape=(encoding_dim,))
#Retrive the last layer of the autoencoder model
decoder_layer=autoencoder.layers[-1]
#create the decoder model
decoder=keras.Model(encoded_input,decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
#scale and make train and test dataset
(X_train,_),(X_test,_)=mnist.load_data()
X_train=X_train.astype('float32')/255.
X_test=X_test.astype('float32')/255.
X_train=X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test=X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)
#train autoencoder with training dataset
autoencoder.fit(X_train,X_train,
 epochs=50,
 batch_size=256,
 shuffle=True,
 validation_data=(X_test,X_test))
encoded_imgs=encoder.predict(X_test)
decoded_imgs=decoder.predict(encoded_imgs)
import matplotlib.pyplot as plt
n = 10 # How many digits we will display
plt.figure(figsize=(40, 4))
for i in range(10):
 # display original
 ax = plt.subplot(3, 20, i + 1)
 plt.imshow(X_test[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 # display encoded image
 ax = plt.subplot(3, 20, i + 1 + 20)
 plt.imshow(encoded_imgs[i].reshape(8,4))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 # display reconstruction
 ax = plt.subplot(3, 20, 2*20 +i+ 1)
 plt.imshow(decoded_imgs[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
plt.show()

# PRACTICAL NO: 09
# Aim: Implementation of convolutional neural network to predict numbers from
# number images.
# Code:


from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

# download mnist data and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# plot the first image in the dataset
plt.imshow(X_train[0])
plt.show()
print(X_train[0].shape)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_train[0]
print(Y_train[0])
model = Sequential()
# add model layers
# learn image features
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
# train
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)
print(model.predict(X_test[:4]))
# actual results for 1st 4 images in the test set
print(Y_test[:4])

# PRACTICAL NO: 10
# Aim: Denoising of images using autoencoder.
# Code:

import keras
from keras.layers import Conv2D, Input
from keras.datasets import mnist
from keras import layers
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
noise_factor = 0.5
X_train_noisy = X_train+noise_factor*np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test+noise_factor*np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
input_img = keras.Input(shape=(28,28,1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_noisy, X_train, epochs=3, batch_size=128, shuffle=True, validation_data=(X_test_noisy, X_test), callbacks=[TensorBoard(log_dir='/tmo/tb', histogram_freq=0, write_graph=False)])
predictions = autoencoder.predict(X_test_noisy)
m = 10
plt.figure(figsize=(20, 2))
for i in range(1, m+1):
    ax = plt.subplot(1, m, i)
    plt.imshow(predictions[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

