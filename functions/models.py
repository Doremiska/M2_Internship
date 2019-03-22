import numpy as np
from keras import backend as K
from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Flatten, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pickle

def build_model(dim_lat, dim_lon, name='cnn'):
       
    # Input = mlp / Output = mlp
    if name in ['mlp']:
        input_shape = (dim_lat*dim_lon,)
        output_shape = dim_lat*dim_lon
    
    # Input = cnn / Output = mlp
    elif name in ['bolton2019']:
        if K.image_data_format() == 'channels_first':
            input_shape = (1, dim_lat, dim_lon)
        else:
            input_shape = (dim_lat, dim_lon, 1)    
        output_shape = dim_lat*dim_lon
    
    # Input = cnn / Output = cnn
    else:
        if K.image_data_format() == 'channels_first':
            input_shape = (1, dim_lat, dim_lon)
        else:
            input_shape = (dim_lat, dim_lon, 1)
    
    inputs = Input(input_shape)
    
    # Model names exemple: unet_3l_64-512f_r
    # - unet: model name
    # - 3l: 3 layers of pool/up
    # - 64-512f: 64 to 512 features/filters
    # - r: with regularization
    
    if name in ['unet_3l_64-512f', 'unet_3l_64-512f_r5e-7', 'unet_3l_64-512f_r1e-6']:
             
        if name == 'unet_3l_64-512f':
            r = 0
        elif name == 'unet_3l_64-512f_r5e-7':
            r = 5e-7
        elif name == 'unet_3l_64-512f_r1e-6':
            r = 1e-6
        
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv4)
        
        up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(r))(UpSampling2D(size=(2,2))(conv4))
        merge5 = concatenate([conv3,up5], axis=3)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(merge5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv5)
        
        up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(r))(UpSampling2D(size=(2,2))(conv5))
        merge6 = concatenate([conv2,up6], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv6)
        
        up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(r))(UpSampling2D(size=(2,2))(conv6))
        merge7 = concatenate([conv1,up7], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv7)
        
        outputs = Conv2D(1, 1)(conv7)
        
    elif name in ['unet_2l_64f_r1e-6', 'unet_2l_32f_r1e-3', 'unet_2l_32f_r1e-6']:
        
        if name == 'unet_2l_64f_r1e-6':
            n = 64
            r = 1e-6
        elif name == 'unet_2l_32f_r1e-3':
            n = 32 
            r = 1e-3 
        elif name == 'unet_2l_32f_r1e-6':
            n = 32
            r = 1e-6
            
        conv1 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(inputs)
        conv1 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(pool1)
        conv2 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(pool2)
        conv3 = Conv2D(n, 3, activation = 'relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv3)
        
        up6 = Conv2D(n, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(r))(UpSampling2D(size=(2,2))(conv3))
        merge6 = concatenate([conv2,up6], axis=3)
        conv6 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(merge6)
        conv6 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv6)
        
        up7 = Conv2D(n, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(r))(UpSampling2D(size=(2,2))(conv6))
        merge7 = concatenate([conv1,up7], axis=3)
        conv7 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(merge7)
        conv7 = Conv2D(n, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(r))(conv7)
        
        outputs = Conv2D(1, 1)(conv7)
        
    elif name in ['bolton2019']:
        conv1 = Conv2D(16, 3, activation='relu', strides=2, padding='valid')(inputs)
        conv1 = Conv2D(8, 3, activation='relu', padding='valid')(conv1)
        conv1 = Conv2D(8, 3, activation='relu', padding='valid')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        flat1 = Flatten()(pool1)
        outputs = Dense(output_shape, activation='linear')(flat1)
         
    elif name == 'cnn':
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        outputs = Conv2D(1, 1)(conv1)
                  
    model = Model(inputs, outputs)
    model.name = name
    
    return model

def plot_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))

def data_to_keras(data, input_shape, output_shape, n_train=40, n_val=5, n_test=5):
    # Split data as training, validation and testing set  
    x_train = data[0:n_train,:,:,:]
    y_train = np.repeat(data[np.newaxis, 50,:,:,:], n_train, axis=0)

    x_val = data[n_train:n_train+n_val,:,:,:]
    y_val = np.repeat(data[np.newaxis, 50,:,:,:], n_val, axis=0)

    x_test = data[n_train+n_val:n_train+n_val+n_test,:,:,:]
    y_test = np.repeat(data[np.newaxis, 50,:,:,:], n_test, axis=0)
    
    # Reshape
    x_train = x_train.reshape((x_train.shape[0]*x_train.shape[1],) + input_shape[1:])
    y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1],) + output_shape[1:])
    
    x_val = x_val.reshape((x_val.shape[0]*x_val.shape[1],) + input_shape[1:])
    y_val = y_val.reshape((y_val.shape[0]*y_val.shape[1],) + output_shape[1:])
    
    x_test = x_test.reshape((x_test.shape[0]*x_test.shape[1],) + input_shape[1:])
    y_test = y_test.reshape((y_test.shape[0]*y_test.shape[1],) + output_shape[1:])
        
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def run_model(model, x_train, y_train, x_val, y_val, lr=1e-4, batch_size=32, epochs=1, patience=30, path=None, save_all=False, save_light=False):
    
    # Callbacks
    checkpointer = ModelCheckpoint(filepath=path+'best_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
    
    # Compile model
    model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
    
    # Run model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[checkpointer, earlystopper, reduce_lr])

    # Save model and history
    if save_all == True:
        save_model(model, path)
        save_history(history, path)
        
    if save_light == True:
        save_weights(model, path)
        save_history_history(history.history, path)
        save_history_params(history.params, path)
        
    return history

def check_path(path):
    temp_path = ''
    for sub_path in path.split('/'):
        temp_path += sub_path+'/'
        if not os.path.isdir(temp_path):
            os.mkdir(temp_path)
        
def save_model(model, path):
    check_path(path)
    model.save(path+'model.h5')
    print(path+'model.h5 saved!')  
    
def get_model(path):
    model = load_model(path+'model.h5')  
    return model

def save_weights(model, path):
    check_path(path)
    model.save_weights(path+'weights.h5')
    print(path+'weights.h5 saved!')
       
def save_history(history, path):
    check_path(path)
    with open(path+'history.p', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    print(path+'history.p saved!')
    
def get_history(path):
    with open(path+'history.p', 'rb') as file_pi:
        history = pickle.load(file_pi)   
    return history

def save_history_history(history_history, path):
    check_path(path)
    with open(path+'history.history.p', 'wb') as file_pi:
        pickle.dump(history_history, file_pi)
    print(path+'history.history.p saved!')
    
def get_history_history(path):
    with open(path+'history.history.p', 'rb') as file_pi:
        history = pickle.load(file_pi)   
    return history

def save_history_params(history_params, path):
    check_path(path)
    with open(path+'history.params.p', 'wb') as file_pi:
        pickle.dump(history_params, file_pi)
    print(path+'history.params.p saved!')
    
def get_history_params(path):
    with open(path+'history.params.p', 'rb') as file_pi:
        params = pickle.load(file_pi)   
    return params
    
