
# coding: utf-8

# In[271]:


from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np


# ## Build The Class

# In[283]:


class ROI(Layer):
    '''
    # Arguments
        pool_size: 7x7 region.
        num_rois: number of regions 
    # Input shape
        X_img:(1, rows, cols, channels)
        X_roi:(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        # Output shape: (1, num_rois,  pool_size, pool_size, channels)
    '''
    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(ROI, self).__init__(**kwargs)

    
    def channels(self, input_shape):
  
               self.nb_channels = input_shape[0][3]
    
    def output_shape(self, input_shape):
  
             return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels


    def compute(self, x):

        assert(len(x) == 2)
        #list of two 4D tensors [X_img,X_roi] with shape:
        #X_img:(1, rows, cols, channels)
        #X_roi: (1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []
        for roi_idx in range(self.num_rois):
            
            x = rois[0, 0]
            y = rois[0, 1]
            w = rois[0, 2]
            h = rois[0, 3]
#             ro_length = tf.multiply(w, 1/(self.pool_size))
# #             ro_length = w / float(self.pool_size)
# #             c_length = h / float(self.pool_size)
#             num_pool_regions = self.pool_size
            #convert tensors into integers 
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size,self.pool_size))
            outputs.append(rs)
            final_output = K.concatenate(outputs, axis=0)
    
        return final_output

    

