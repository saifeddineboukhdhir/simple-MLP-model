
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback

import numpy as np 
import random


# get data 
list_xy=[([1.5,3],0),([1.5,7],0),([2.5,3.9],0),([2.5,9],0),([3.5,2.9],0),([3.5,7],0),([4,2],0),([4,9],0),
                ([6,5],0),([6.5,3.9],0),([6.5,8],0),([7,3],0),([7,6],0),([7.5,4],0),([8,1],0),([8.5,9],0),
                
                ([2,5],1),([2.5,6],1),([3,5],1),([4.5,5],1),([7.5,7],1),([8,5],1),([8,8],1),([8.5,2.9],1),([8.5,7],1),
                ([9,4],1),([9,6],1),([9.5,4.9],1),([9.5,8],1)
               ]
random.shuffle(list_xy)
x_data=np.array([a[0] for a in list_xy])
y_data=np.array([[a[1]] for a in list_xy])

# build model 


model = Sequential()
model.add(Dense(30, activation="sigmoid", input_dim=2))
#model.add(Dropout(0.1))
model.add(Dense(16, activation="linear"))
model.add(Dense(2, activation="tanh"))
model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=['accuracy'])


# Records the weights throughout the training process
weights_history = []
bias_history = []
class MyCallback(Callback):
    def on_batch_end(self, batch, logs):
        weights1, biases1 = model.layers[0].get_weights()
        weights2, biases2 = model.layers[1].get_weights()
        weights3, biases3 = model.layers[2].get_weights()
        weights = [weights1, weights2,weights3]
        biases = [biases1, biases2,biases3]
        #print('on_batch_end() model.weights:', weights)
        weights_history.append(weights)
        bias_history.append(biases)


callback = MyCallback()

# split data 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.0, shuffle=True)
y_train= to_categorical(y_train)
#y_test= to_categorical(y_test)


model.fit(X_train,  y_train,
           epochs=100, batch_size=1,callbacks=[callback])


summary = model.summary()
# the first four epochs 
for iteration in range(4):
    print("Iteration:  ", iteration+1,"=========================================\n" )
    print("Weight layer 1 \n")
    print('W1=   ',weights_history[iteration][0],"\n")
    print("Bias layer 1 \n ")
    print("b1=   ", bias_history[iteration][0],"\n")
    print("Weight layer 2 \n")
    print('W2=   ',weights_history[iteration][1],"\n")
    print("Bias layer 2 \n ")
    print("b2=   ", bias_history[iteration][1],"\n")
    print("Weight layer 3 \n")
    print('W3=   ', weights_history[iteration][2],"\n")
    print("Bias layer 3  \n ")
    print("b3=   ",bias_history[iteration][2],"\n")

from sklearn.metrics import confusion_matrix
# confusion matrix 
predictions = model.predict(X_train)
y_train = np.argmax(y_train, axis=-1)
predictions = np.argmax(predictions, axis=-1)
c = confusion_matrix(y_train, predictions)
print('Confusion matrix:\n', c)
print('sensitivity', c[0, 0] / (c[0, 1] + c[0, 0]))
print('specificity', c[1, 1] / (c[1, 1] + c[1, 0]))
