import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def DirectionNetwork(pretrained_weights = None,input_size = (512,512,2),loss="direction"):
    inputs  = Input(input_size)
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    pool1   = pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
    #The second conv layer set
    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
    pool2   = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
    #The third layer set
    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
    conv3_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_2)
    pool3   = AveragePooling2D(pool_size=(2, 2))(conv3_3)
    
    #The fourth layer set
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    pool4   = AveragePooling2D(pool_size=(2, 2))(conv4_3)
    
    #The fifth layer set
    conv5_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
    conv5_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_2)
    print("Built all the CNN Layers 1 to 5")
    
    #The FCNs 
    #The conv5 layers
    fcn5_1  = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_3)
    fcn5_2  = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn5_1)
    fcn5_3  = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn5_2)
    
    #The conv4 layers
    fcn4_1  = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    fcn4_2  = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn4_1)
    fcn4_3  = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn4_2)
    
    #The conv3 layers
    fcn3_1  = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_3)
    fcn3_2  = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn3_1)
    fcn3_3  = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fcn3_2)
    print("Built all FCN Layers")
    
    #The upscore layers
    upscr5_3= Conv2DTranspose(256,kernel_size=8,strides=4,padding = 'same')(fcn5_3)
    upscr4_3= Conv2DTranspose(256,kernel_size=4,strides=2,padding = 'same')(fcn4_3)
    
    fuse3   = merge([fcn3_3,upscr5_3,upscr4_3], mode = 'concat', concat_axis = 3)
    fuse3_1 = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fuse3)
    fuse3_2 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(fuse3_1) 
    fuse3_3 = Conv2D(2,   1, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(fuse3_2) # Output channels=2
    output  = Conv2DTranspose(2,kernel_size=16,strides=16,padding = 'same')(fcn5_3)
    model = Model(input = inputs, output = output)

    def inversecosdistance(y_true,y_pred):
        weight = tf.expand_dims(y_true[:,:,:,2],axis=-1)
        ss=tf.expand_dims(y_true[:,:,:,3],axis=-1)
        ss=tf.to_float(tf.reshape(ss, (-1, 1)))
        y_true=tf.stack([y_true[:,:,:,0],y_true[:,:,:,1]],axis=-1)
        y_pred = tf.reshape(y_pred, (-1,2))
        y_true = tf.to_float(tf.reshape(y_true, (-1,2)))
        weight = tf.to_float(tf.reshape(weight, (-1, 1)))
        pred = tf.nn.l2_normalize(y_pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(y_true, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True))
        lossAngleTotal = tf.reduce_sum((tf.abs(errorAngles*errorAngles)))

        return lossAngleTotal

    model.summary()
    model.compile(optimizer = Adam(lr = 1e-6), loss = inversecosdistance, metrics = ['accuracy'])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model