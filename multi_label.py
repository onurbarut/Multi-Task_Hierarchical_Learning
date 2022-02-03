import json
import argparse
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, CuDNNLSTM, concatenate, Flatten, Conv1D, GlobalMaxPooling1D

from utils.helper2 import *
def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def createModel(X_train, OUTPUT, filters=250, kernel_size=3, dropout_rate=0., CNN_layers=2, clf_reg=1e-4):
    # Model Definition
    OUTPUTS = []
    #raw_inputs = Input(shape=(X_train.shape[1]))
    raw_inputs = Input(shape=(X_train.shape[1],))
    xcnn = Embedding(1000, 50, input_length=121)(raw_inputs)  
    xcnn = Conv1D(filters, (kernel_size), padding='valid', activation='relu', strides=1)(xcnn)    
    if dropout_rate != 0:
        xcnn = Dropout(dropout_rate)(xcnn)

    for i in range(1, CNN_layers-1):
        xcnn = Conv1D(filters,
                    (kernel_size),
                    padding='valid',
                    activation='relu',
                    strides=1)(raw_inputs)
    if dropout_rate != 0:
        xcnn = Dropout(dropout_rate)(xcnn)  

    # we use max pooling:
    xcnn = GlobalMaxPooling1D()(xcnn)

    if 'top' in OUTPUT: 
        top_level_predictions = Dense(OUTPUT['top'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xcnn)
    OUTPUTS.append(top_level_predictions)

    if 'mid' in OUTPUT: 
        mid_level_predictions = Dense(OUTPUT['mid'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='mid_level_output')(xcnn)
    OUTPUTS.append(mid_level_predictions)

    if 'fine' in OUTPUT: 
        fine_grained_predictions = Dense(OUTPUT['fine'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='fine_grained_output')(xcnn)
    OUTPUTS.append(fine_grained_predictions)    

    model = Model(inputs=raw_inputs, outputs=OUTPUTS)

    return model

dataset = "./data/NetML" # or "./data/CICIDS2017" or "./data/non-vpn2016"
anno = "top" # or "mid" or "fine"
#submit = "both" # or "test-std" or "test-challenge"

# Assign variables
training_set = dataset+"/2_training_set"
training_anno_file = dataset+"/2_training_annotations/2_training_anno_"+anno+".json.gz"
test_set = dataset+"/1_test-std_set"
challenge_set = dataset+"/0_test-challenge_set"


# Create folder for the results
time_ = t.strftime("%Y%m%d-%H%M%S")
save_dir = os.getcwd() + '/results/' + time_
os.makedirs(save_dir)

# Get training data in np.array format
#Xtrain, ytrain, class_label_pair, _ = get_training_data(training_set, training_anno_file)
annotationFileName = [dataset+"/2_training_annotations/2_training_anno_top.json.gz", dataset+"/2_training_annotations/2_training_anno_fine.json.gz"]
feature_names, ids, Xtrain, label_list, class_label_pairs_list = read_dataset(training_set, annotationFileName=annotationFileName, class_label_pairs=None)

# Split validation set from training data
X_train, X_val, y_train, y_val = train_test_split(Xtrain, np.transpose(np.asarray(label_list)),
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=np.transpose(np.asarray(label_list)))

# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Get name of each class to display in confusion matrix
top_class_names = list(sorted(class_label_pairs_list[0].keys()))
fine_class_names = list(sorted(class_label_pairs_list[1].keys()))


# Get model
# Default Training Hyperparameters

# Create folder for the results
time_ = t.strftime("%Y%m%d-%H%M%S")
save_dir = os.getcwd() + '/results/' + time_
os.makedirs(save_dir)

n_classes_top = len(top_class_names)
n_classes_fine = len(fine_class_names)
learning_rate = 1e-5
decay_rate = 1e-5
dropout_rate = 0.5
n_batch = 300
n_epochs = 200  # Loop 500 times on the dataset
filters = 128
kernel_size = 3
CNN_layers = 2
loss_weights = [1., 1.]
clf_reg = 1e-5

#model = createModel(X_train, {'top':n_classes}, filters=filters, kernel_size=kernel_size, dropout_rate=dropout_rate, CNN_layers=CNN_layers, clf_reg=clf_reg)
"""
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
"""
# Model Definition
OUTPUT = {}
OUTPUT['top'] = n_classes_top
OUTPUT['fine'] = n_classes_fine
OUTPUTS = []
raw_inputs = Input(shape=(X_train.shape[1],))
xdense = Dense(256, activation='softmax',name='hidden_layer')(raw_inputs)
xdense = Dropout(0.5)(xdense)

top_level_predictions = Dense(OUTPUT['top'], activation='softmax', 
        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
        name='top_level_output')(xdense)
OUTPUTS.append(top_level_predictions)

fine_grained_predictions = Dense(OUTPUT['fine'], activation='softmax', 
        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
        name='fine_grained_output')(xdense)
OUTPUTS.append(fine_grained_predictions)    

model = Model(inputs=raw_inputs, outputs=OUTPUTS)
print(model.summary()) # summarize layers
plot_model(model, to_file=save_dir+'/model.png') # plot graph
model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
  optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
  loss_weights=loss_weights,
  metrics=['accuracy'])
# Train the model
history=model.fit(X_train_scaled, [one_hot(y_train[:,0], n_classes_top), one_hot(y_train[:,1], n_classes_fine),], batch_size=n_batch, epochs=n_epochs, validation_data=(X_val_scaled, [one_hot(y_val[:,0], n_classes_top), one_hot(y_val[:,1], n_classes_fine)]))

# Output accuracy of classifier
print("Training Score: \t{:.5f}".format(history.history['top_level_output_acc'][-1]))
print("Training Score: \t{:.5f}".format(history.history['fine_grained_output_acc'][-1]))
print("Validation Score: \t{:.5f}".format(history.history['val_top_level_output_acc'][-1]))
print("Validation Score: \t{:.5f}".format(history.history['val_fine_grained_output_acc'][-1]))

# Print Confusion Matrix
ypred = model.predict(X_val_scaled)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(directory=save_dir, y_true=y_val[:,0], y_pred=ypred[0].argmax(1), 
                        classes=top_class_names, 
                        normalize=False)
plot_confusion_matrix(directory=save_dir, y_true=y_val[:,1], y_pred=ypred[1].argmax(1), 
                        classes=fine_class_names, 
                        normalize=False)

for k, v in history.history.items():
        if k == 'top_level_output_acc':
            x1 = 'train_top_accuracy'
            y1 = v
        elif k == 'fine_grained_output_acc':
            x1f = 'train_fine_accuracy'
            y1f = v
        elif k == 'val_top_level_output_acc':
            x2 = 'validation_top_accuracy'
            y2 = v
        elif k == 'val_fine_grained_output_acc':
            x2f = 'validation_fine_accuracy'
            y2f = v
        elif k == 'top_level_output_loss':
            x3 = 'train_top_loss'
            y3 = v
        elif k == 'fine_grained_output_loss':
            x3f = 'train_fine_loss'
            y3f = v
        elif k == 'val_top_level_output_loss':
            x4 = 'validation_top_loss'
            y4 = v
        elif k == 'val_fine_grained_output_loss':
            x4f = 'validation_fine_loss'
            y4f = v

plt.figure()  
plt.plot(y1, 'r-')
plt.plot(y1f, 'r--')
plt.plot(y2, 'b-')
plt.plot(y2f, 'b--')
plt.title('model classification accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([x1, x1f, x2, x2f], loc='best')
plt.savefig(save_dir+'/accuracy.png')

plt.figure()  
plt.plot(y3, 'r-')
plt.plot(y3f, 'r--')
plt.plot(y4, 'b-')
plt.plot(y4f, 'b--')
plt.title('model classification loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([x3, x3f, x4, x4f], loc='best')
plt.savefig(save_dir+'/loss.png')