
# coding: utf-8

# [View in Colaboratory](https://colab.research.google.com/github/sameher1/First_Project/blob/master/VGG16.ipynb)

# # **Import modules **

# In[1]:


try:
    import ROI
except ImportError:
    import sys
    sys.path.append('local_modules')
    import ROI
import tensorflow as tf
from keras.models import Model
from keras.layers import  Conv2D,Dense,Flatten,MaxPool2D,Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from __future__ import print_function
from __future__ import absolute_import
import warnings
from keras import backend as K
'''
There are two ways to build models in keras: Sequential and functional. Sequential ===>sequential and functional====>Model
'''


# # **Get VGG16 weights **

# In[2]:


def path_to_get_weight(path):
  '''
  vgg16_weights_tf_dim_ordering_tf_kernels.h5
  Json file to save the model description 
  HDF (.h5) to save the weights 
  '''
  return path


# # **Build the CNN model **

# In[3]:


def CNN(input_tensor=None,trainable=False):
  # Handle input with random size, multiple inputs, inputs which are not tensors 
  input_shape = (None, None,3)


  if input_tensor is None:
        input_img = Input(shape=input_shape)
  else:
        if not K.is_keras_tensor(input_tensor):
            input_img = Input(tensor=input_tensor, shape=input_shape)
        else:
            input_img = input_tensor

  nbre_channel=3
  ########################################################################
  
  #Block 1
  x=Conv2D(64,(3,3),activation='relu',padding='same',name='bcnn1')(input_img)
  x=Conv2D(64,(3,3),activation='relu',padding='same',name='bcnn2')(x)
  x=MaxPool2D((2,2), strides=(2,2),name='pool1')(x)
  
  #Block 2
  x=Conv2D(128,(3,3),activation='relu',padding='same',name='b2cnn1')(x)
  x=Conv2D(128,(3,3),activation='relu',padding='same',name='b2cnn2')(x)
  x=MaxPool2D((2,2), strides=(2,2),name='pool2')(x)
  
  
  #Block 3
  x=Conv2D(256,(3,3),activation='relu',padding='same',name='b3cnn1')(x)
  x=Conv2D(256,(3,3),activation='relu',padding='same',name='b3cnn2')(x)
  x=Conv2D(256,(3,3),activation='relu',padding='same',name='b3cnn3')(x)
  x=MaxPool2D((2,2), strides=(2,2),name='pool3')(x)
  
  
  #Block 4
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b4cnn1')(x)
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b4cnn2')(x)
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b4cnn3')(x)
  x=MaxPool2D((2,2), strides=(2,2),name='pool4')(x)
  
  #Block 5
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b5cnn1')(x)
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b5cnn2')(x)
  x=Conv2D(512,(3,3),activation='relu',padding='same',name='b5cnn3')(x)
  #x=MaxPool2D((2,2), strides=(2,2),name='pool4')(x)
  
  return x

  


# # **RPN**

# In[4]:


def RPN(cnn_layer,nbr_anchors):
  
  x=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='normal',name='RPN1')(cnn_layer)
  xc=Conv2D(nbr_anchors,(1,1),activation='sigmoid',kernel_initializer='uniform',name='classification')(x)
  xr=Conv2D(4*nbr_anchors,(1,1),activation='linear',kernel_initializer='zero',name='regression')(x)
  
  return [xc,xr,cnn_layer]


  
  


# # **Classifier **

# In[5]:


def classification(cnn_layer,input_rois, num_rois,nbre_class=3, trainable= False):
   input_shape=(num_rois,(7,7),512)
   
   output_roi= ROI(7,num_rois)([cnn_layer,input_rois])
   '''
   Dense implements the operation: output = activation(dot(input, kernel) + bias)
   TimeDistributed can be used with arbitrary layers, not just Dense, for instance with a Conv2D layer
   '''
   out = TimeDistributed(Flatten(name='flatten'))(output_roi)
   out = TimeDistributed(Dense(4096, activation='relu', name= 'FC1'))(out)
   out = TimeDistributed(Dense(4096, activation='relu', name= 'FC2'))(out)
   
   outc = TimeDistributed(Dense(nbre_class, activation='softmax',  kernel_initializer='zero', name= 'FCC'))(out)
   outr=TimeDistributed(Dense(4*(nbre_class-1), activation='linear',  kernel_initializer='zero', name= 'FCL'))(out)
   return [outc, outr]
   
   

