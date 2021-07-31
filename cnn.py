import keras

def create_model(SIZE):
    INPUT_SHAPE = (SIZE, SIZE, 3)
    
    inp = keras.layers.Input(shape=INPUT_SHAPE)
    
    conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
    drop1 = keras.layers.Dropout(rate=0.2)(norm1)
    
    flat = keras.layers.Flatten()(drop1)
    
    hidden1 = keras.layers.Dense(512, activation='relu')(flat)
    norm2 = keras.layers.BatchNormalization(axis = -1)(hidden1)
    hidden2 = keras.layers.Dense(256, activation='relu')(norm2)
    norm3 = keras.layers.BatchNormalization(axis = -1)(hidden2)
    drop2 = keras.layers.Dropout(rate=0.2)(norm3)
    
    out = keras.layers.Dense(2, activation='sigmoid')(drop2)
    
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    print(model.summary())
    return model
