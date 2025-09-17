import numpy as np
import tensorflow as tf
import random
import h5py
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)
from sklearn.metrics import confusion_matrix
from Dataset import TripartiteData, QuadripartiteData, TripartiteCatData, QuadripartiteCatData
from QDA import TripartiteQDA, QuadripartiteQDA
from sharedCNN import TripartiteModel, QuadripartiteModel
from tensorflow.keras.callbacks import EarlyStopping

'''-----------------Tripartite cases-----------------'''
# xtr1, ytr1, xt1, yt1 = TripartiteData(num_states=15000, include_mc=False)
# xtr2, ytr2, xt2, yt2 = TripartiteCatData(num_states=15000)
# xtrain = np.concatenate([xtr1, xtr2]); ytrain = np.concatenate([ytr1, ytr2])
# xtest  = np.concatenate([xt1,  xt2 ]); ytest  = np.concatenate([yt1,  yt2 ])
# # xtrain, ytrain, xtest, ytest = TripartiteData(num_states=30000, include_mc=False)
# # xtrain, ytrain, xtest, ytest = TripartiteCatData(num_states=15000)
# print("The number of samples before QDA:" + str(xtrain.shape[0]))

# xtrain, ytrain = TripartiteQDA(xtrain, ytrain) # Quantum Data Augmentation (QDA) for tripartite states
# print("The number of samples after QDA:" + str(xtrain.shape[0]))

# model = TripartiteModel()

'''-----------------Quadripartite cases-----------------'''
xtr1, ytr1, xt1, yt1 = QuadripartiteData(num_states=10000)
xtr2, ytr2, xt2, yt2 = QuadripartiteCatData(num_states=10000)
xtrain = np.concatenate([xtr1, xtr2]); ytrain = np.concatenate([ytr1, ytr2])
xtest  = np.concatenate([xt1,  xt2 ]); ytest  = np.concatenate([yt1,  yt2 ])
# xtrain, ytrain, xtest, ytest = QuadripartiteData(num_states=25000)
# xtrain, ytrain, xtest, ytest = QuadripartiteCatData(num_states=15000)
print("The number of samples before QDA:" + str(xtrain.shape[0]))

# xtrain, ytrain = QuadripartiteQDA(xtrain, ytrain) # Quantum Data Augmentation (QDA) for quadripartite states
# print("The number of samples after QDA:" + str(xtrain.shape[0]))

model = QuadripartiteModel()

'''-----------------Training-----------------'''
Nepochs = 2000
Nbatch = 256

adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=50,         
    verbose=1,           
    mode='min',          
    restore_best_weights=True 
)

with tf.device('/gpu:0'):
    history = model.fit( 
        # for tripartite states
        # [xtrain[:, 0:4, :, :].transpose(0, 2, 3, 1), xtrain[:, 4:8, :, :].transpose(0, 2, 3, 1), xtrain[:, 8:12, :, :].transpose(0, 2, 3, 1)], ytrain,   

        # for tripartite states
        [xtrain[:, 0:4, :, :].transpose(0, 2, 3, 1), xtrain[:, 4:8, :, :].transpose(0, 2, 3, 1), xtrain[:, 8:12, :, :].transpose(0, 2, 3, 1), xtrain[:, 12:16, :, :].transpose(0, 2, 3, 1)], ytrain,   
        batch_size = Nbatch, 
        epochs = Nepochs, 
        validation_split = 0.3,
        callbacks=[early_stopping])

'''-----------------Testing-----------------'''
# for tripartite states
# ypred = model.predict([xtest[:, 0:4, :, :].transpose(0, 2, 3, 1), xtest[:, 4:8, :, :].transpose(0, 2, 3, 1), xtest[:, 8:12, :, :].transpose(0, 2, 3, 1)])   

# for quadripartite states
ypred = model.predict([xtrain[:, 0:4, :, :].transpose(0, 2, 3, 1), xtrain[:, 4:8, :, :].transpose(0, 2, 3, 1), xtrain[:, 8:12, :, :].transpose(0, 2, 3, 1), xtrain[:, 12:16, :, :].transpose(0, 2, 3, 1)])

y_true_labels = np.argmax(ytest, axis=1)
y_pred_labels = np.argmax(ypred, axis=1)
accuracy = np.mean(y_true_labels == y_pred_labels)
print(f"accuracy:",accuracy)

cm = confusion_matrix(y_true_labels, y_pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(f"Confusion Matrix:",cm_normalized)

