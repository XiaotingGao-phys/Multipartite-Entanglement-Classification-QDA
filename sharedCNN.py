import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model

def shared_sub_model(input_shape):
    model = tf.keras.models.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.7),
        Dense(256, activation='relu'),
        Dropout(0.7),
        Dense(8, activation='relu')
    ])
    return model

def TripartiteModel():
    shared_model = shared_sub_model([24, 24, 4])
    
    input1 = Input(shape=[24, 24, 4])
    input2 = Input(shape=[24, 24, 4])
    input3 = Input(shape=[24, 24, 4])
    output1 = shared_model(input1)
    output2 = shared_model(input2)
    output3 = shared_model(input3)
    
    concatenated = Concatenate()([output1, output2, output3])
    dense1 = Dense(units=8, activation='relu')(concatenated)
    dense2 = Dense(units=3, activation='relu')(dense1)
    dense3 = Dense(units=3, activation='softmax')(dense2)
    
    model = Model(inputs=[input1, input2, input3], outputs=dense3)
    return model

def QuadripartiteModel():
    shared_model = shared_sub_model([24, 24, 4])

    input1 = Input(shape=[24, 24, 4])
    input2 = Input(shape=[24, 24, 4])
    input3 = Input(shape=[24, 24, 4])
    input4 = Input(shape=[24, 24, 4])

    output1 = shared_model(input1)
    output2 = shared_model(input2)
    output3 = shared_model(input3)
    output4 = shared_model(input4)

    concatenated = Concatenate()([output1, output2, output3, output4])

    dense1 = Dense(units=16, activation='relu')(concatenated)
    dense2 = Dense(units=8, activation='relu')(dense1)
    dense3 = Dense(units=5, activation='softmax')(dense2)

    model = Model(inputs=[input1, input2, input3, input4], outputs=dense3)
    return model