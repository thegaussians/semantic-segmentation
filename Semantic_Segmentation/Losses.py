from keras.backend import tensorflow_backend as K

class focal_loss:

    """ A loss function similar to cross_entropy

        # Usage
            model.compile('sgd',loss=focal_loss.loss,.......)

        # Arguments
            class_weights : weights for each class to solve the class imbalance problem.
                            dtype --> array   default --> None
            pixel_weights : weights for each pixels in order to segment certain part of the image clearly.
                            dtype --> array   default --> None
    """
    c_weights = lambda self,x: 1 if x==None else x
    p_weights = lambda self,x: 1 if x==None else x
    clipping = lambda self,x: K.clip(x, K.epsilon(), 1.-K.epsilon())


    def __init__(self,class_weights=None, pixel_weights=None, gamma=2):
        self.class_weights = class_weights
        self.gamma = gamma
        self.pixel_weights = pixel_weights


    def loss(self,y_true,y_pred):

        """ executes the focal loss

            # Arguments
                y_true : true class values
                y_pred : predicted class values from the model
            # Returns
                fl : mean focal loss for the given batch
         """
        y_pred = self.clipping(y_pred)
        fl = -( self.p_weights(self.pixel_weights) * (self.c_weights(self.class_weights) * 0.25 * K.pow(1.-y_pred,self.gamma) * (y_true * K.log(y_pred))) )
        fl = K.sum(fl,axis=(1,2,3))
        fl = K.mean(fl, axis=0)
        return fl/100           ##since the loss is sum over the spatial dimensions it's scale will be high, thus we scale down by 100 to prevent higher gradients



class cross_entropy(focal_loss):

    """ Categorical cross_entropy
        NOTE : for binary classification it uses softmax instead sigmoid

        # Usage
            model.compile('sgd',loss=cross_entropy.loss,.......)

        # Arguments
            class_weights : weights for each class to solve the class imbalance problem.
                            dtype --> array   default --> None
            pixel_weights : weights for each pixels in order to segment certain part of the image clearly.
                            dtype --> array   default --> None
    """

    ## NOTE - this class inherits the properties of focal_loss class
    def loss(self,y_true,y_pred):

        """ executes the categorical cross-entropy

            # Arguments
                y_true : true class values
                y_pred : predicted class values from the model
            # Returns
                ce : mean cross-entropy for the given batch
        """
        y_pred = super().clipping(y_pred)
        ce = -( super().p_weights(self.pixel_weights) * (super().c_weights(self.class_weights) * (y_true * K.log(y_pred))) )
        ce = K.sum(ce,axis=(1,2,3))
        ce = K.mean(ce,axis=0)
        return ce/100               ##since the loss is sum over the spatial dimensions it's scale will be high, thus we scale down by 100 to prevent higher gradients




class dice_loss(focal_loss):

    """ Its similar to IoU, dice_coeff = (2*A^B)/A U B  dice_loss= 1- dice_coeff
        # Usage
            model.compile('sgd',loss=dice_loss.loss,.......)

        # Arguments
            class_weights : weights for each class to solve the class imbalance problem.
                            dtype --> array   default --> None
            pixel_weights : weights for each pixels in order to segment certain part of the image clearly.
                            dtype --> array   default --> None
    """

    ## NOTE - this class inherits the properties of focal_loss class
    def loss(self,y_true,y_pred):

        """ executes the dice loss

            # Arguments
                y_true : true class values
                y_pred : predicted class values from the model
            # Returns
                dl : dice loss for the given batch
        """
        y_pred = super().clipping(y_pred)
        intersection = K.sum((y_true * y_pred),axis=(0,1,2,3))
        union = K.sum((y_true*y_true) + (y_pred*y_pred),axis=(0,1,2,3))
        dl = 1. - ((2*intersection)/union)
        return dl

