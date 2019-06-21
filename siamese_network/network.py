'''

'''


import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,Activation
from keras import backend as K
from keras.optimizers import Adadelta
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


epochs = 100

#------------------------------------------------------------------------------
#   prepare dataset
#------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x = np.concatenate((x_train,x_test), axis = 0)
y = np.concatenate((y_train, y_test), axis = 0)
#data A has digit with 2,3,4,5,6,7  and data B has digit 0,1,8,9
mask_A = np.isin(y,[2,3,4,5,6,7])
mask_B  = np.isin(y,[0,1,8,9])


A = x[mask_A]
B = x[mask_B]

label_A = y[mask_A]
label_B = y[mask_B]
# split into training 0.8 and testing 0.2

x_train, x_test, y_train, y_test = train_test_split(A, label_A,test_size = 0.2)

#dataset used to evl
# both [2,3,4,5,6,7] and [0,1,8,9]
union_x = np.concatenate((x_test, B), axis = 0)
union_y = np.concatenate((y_test, label_B), axis = 0)

union_x = union_x.astype('float32')
union_x /= 255

#data set for testing [0,1,8,9]
x_B = B.astype('float32')
x_B /= 255

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

#------------------------------------------------------------------------------
#   loss function
#------------------------------------------------------------------------------
def contrastive_loss(y_true, y_pred):

    margin = 1
    sq_pred = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0))
    loss =  K.mean(y_true * sq_pred + (1 - y_true) * margin_sq)
    return loss
#------------------------------------------------------------------------------
#   Create shared network
#------------------------------------------------------------------------------
  
def shared_network(input_shape):

    input = Input(shape=input_shape)
    layer = Flatten()(input)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128, activation='relu')(layer)
    return Model(input, layer)


#def shared_network(input_shape):
#    input = Input(shape = input_shape)
   
#    layer = Conv2D(8, kernel_size=(3, 3),
#                    activation='relu', input_shape = (input_shape)
#                      )(input)
#    layer = Conv2D(64, (3, 3), activation='relu')(layer)
#    layer = MaxPooling2D(pool_size=(2, 2))(layer)
#    layer = Dropout(0.25)(layer)
#    layer = Flatten()(layer)
#    layer = Dense(128, activation='relu')(layer)
#    layer = Dropout(0.5)(layer)
#    layer = Dense(6, activation='relu')(layer)
#    return Model(input,layer)
    
    

# input pair data into network

siamese = shared_network(input_shape)
input_left = Input(shape=input_shape)
input_right = Input(shape=input_shape)


output_left = siamese(input_left)
output_right = siamese(input_right)
#------------------------------------------------------------------------------
#   create pos and neg pairs for networks
#------------------------------------------------------------------------------
def create_pairs(x, digit_indices):

    pairs = []
    labels = []
    n = min([len(digit_indices[j]) for j in range(num_classes)]) - 1
    for j in range(num_classes):
        for i in range(n):
            p1, p2 = digit_indices[j][i], digit_indices[j][i + 1]
            pairs += [[x[p1], x[p2]]]
            inc = random.randrange(1, num_classes)
            jn = (j + inc) % num_classes
            p1, p2 = digit_indices[j][i], digit_indices[jn][i]
            pairs += [[x[p1], x[p2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

# digits only select from 2,3,4,5,6,7
num_classes = len(np.unique(y_train))   

#range 2 - 8 is 2,3,4,5,6,7,
digit_indices = [np.where(y_train == i)[0] for i in range(2,8)]
train_pairs, train_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(2,8)]
test_pairs, test_y = create_pairs(x_test, digit_indices)


#------------------------------------------------------------------------------
#   find eucliden distance between two 
#------------------------------------------------------------------------------
def euclidean_distance(vects):
    x, y = vects
    sum_sq = K.sum(K.square(x - y), axis=1, keepdims=True)
    distance = K.sqrt(K.maximum(sum_sq, K.epsilon()))
    return distance


def eucl_shape(shape):
    shape1, shape2 = shape
    return (shape1[0], 1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_shape)([output_left, output_right])


#------------------------------------------------------------------------------
#   build model and train
#------------------------------------------------------------------------------
model = Model([input_left, input_right], distance)
ada = Adadelta()


threshold = 0.5
# fixed threshold on distances
def acc(y_true, y_pred):

    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))






model.compile(loss=contrastive_loss, optimizer=ada, metrics=[acc])


#save the best weights to load 
filepath="model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'atuo')

#early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'auto' )

history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
          batch_size=128,
          epochs=epochs,
          callbacks = [checkpoint],
          validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))



#load best weight for testing
model.load_weights("model.hdf5")


def compute_accuracy(y_true, y_pred):
    
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)






#------------------------------------------------------------------------------
# plot for report
#------------------------------------------------------------------------------
print(history.history.keys())
#accurancy
plt.rcParams['figure.figsize'] = [7,7]
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc = 'upper left')
plt.show()
# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()





#------------------------------------------------------------------------------
#  Results for [2,3,4,5,6,7]
#------------------------------------------------------------------------------
y_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
train_acc = compute_accuracy(train_y, y_pred)

y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
test_acc = compute_accuracy(test_y, y_pred)




print(' Accuracy on training set:',train_acc)
print('Accuracy on test set:', test_acc)



#------------------------------------------------------------------------------
#  evaluating with [0,1,8,9] only 
#------------------------------------------------------------------------------

input_shape = x_B.shape[1:]
num_classes = len(np.unique(label_B))
#only select 0,1,8,9
digit_indices = [np.where(label_B == i)[0] for i in range(2)]
digit_indices += [np.where(label_B == i)[0] for i in range(8,10)]
train_pairs_B, train_y_B = create_pairs(x_B, digit_indices)

y_pred = model.predict([train_pairs_B[:,0],train_pairs_B[:,1]])
dataB = compute_accuracy(train_y_B, y_pred)
print('Accuracy on unseen set:',  dataB)



#------------------------------------------------------------------------------
#  evaluating with [0,1,8,9] and [2,3,4,5,6,7]
#------------------------------------------------------------------------------
input_shape = union_x.shape[1:]
num_classes = len(np.unique(union_y))

digit_indices = [np.where(union_y == i)[0] for i in range(num_classes)]
train_pairs_union, train_y_union = create_pairs(union_x, digit_indices)

y_pred = model.predict([train_pairs_union[:,0], train_pairs_union[:,1]])
union_scores = compute_accuracy(train_y_union, y_pred)
print('Accuracy on union set:',union_scores)


