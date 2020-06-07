# Semantic Segmentation Package in Keras



# News
 This project is under construction(much more functionalities to be added), but you can use the available fucntions. I always welcome and highly appreciate contributions and suggestions from the user. Feel free to contact me at --> pratoshk11@gmail.com 
 
 
## Updates - 

 (6/07) --> There was some bugs in the previous version of this package, I fixed it compeletely now its working fine, my apologies to the cloners.
 
 (6/07) --> Added a new model - Deep residual Unet, added the functionality to handle segmented data.

 (4/24) --> Fixed a bug regarding pixel_weights and now it works fine.      
  
 (4/24) --> A new metric 'Dice Coefficient' has been added, it measures the overlap score b/w images  DC = (2*A^B)/A U B. 
  



# Description

This repository serves as a Semantic Segmentation package. The motive is to ease the steps of implementing, training, and testing new Semantic Segmentation models, following are the overall functionalities:
  
  * easily building model architectures, pre-trained models also available
  * getting raw image data ready for segmentation
  * training and testing models
  * various loss functions- CCE,focal-loss,dice-loss,GDL,Tversky-loss,etc
  * metrics like IoU,recall,precision,f1score
  * visualising the loss functions and metrics over epochs 
 
 
 # Current version
 
 ## Models
 
 * FCN32 - Uses a pre-trained CNN model as an encoder and upsamples the image in the decoder part using transposed convolution.
   Since it upscales the image by 32 its called FCN32. Available - FCN32-vgg16, FCN32-vgg19,FCN32-resnet50      
 
 * FCN16 - Similar to FCN32 architecture, it has the same encoder as FCN32 but uses a skip connection in decoder to get accurate output.
   After skip connection it upsamples by scale of 16 hence the name FCN16. Available - FCN16-vgg16, FCN16-vgg19,FCN16-resnet50
 
 * FCN8 - Similar to FCN16 architecture, it uses 2 skip connections to get much finer outputs and after skip connections it upsamples by    scale of 8 hence FCN8. Available - FCN8-vgg16, FCN8-vgg19,FCN8-resnet50
 
 * UNET - Uses small filters( 3*3 throught) to extract low-level features and results much finer outputs, since they are using multiple skip connections across layers.
 
 * ResUnet - This is state-of-the-art network for segmentation(if hyperparameters are tuned properly), its combination of unet and residual networks which makes it further powerful.
 
 
 
 ## Loss Functions  
 
 Categorical Cross Entropy, Focal-loss, Dice-loss 
 
 All the above loss functions comes with class weights to prevent class imbalance and region weights(pixel weights) to concentrate on
 certain parts of the image.
 
 NOTE - The user has to provide the weights matching the dimension
 
 
 ## Metrics
 
 Intersection over Union, accuracy, recall, precision, f1score, Dice Coefficient
 
 All the above mentioned metrics comes with class weights to calculate weighted average scores among the classes for imbalanced datasets.
 
 
 
 
# Usage (steps for current version)

 You can start by cloning this repository to your project folder(the folder you are working)
 
 ## For building the model
    from Semantic_Segmentation import models                               ---> import the module
    fcn = models.FCN.fcn32(input_shape,n_classes,'vgg19',regularizer=None,summary=True)     ---> initiate your requirements
    model = fcn.build()                                                     ---> executes building process and returns the architecture
    
 ## For loss function
    from Semantic_Segmentation.Losses import focal_loss
    fl = focal_loss.loss(y_true,y_pred,class_weights,pixel_weights)
    ''' Now you can use this loss function '''
    
 ## For metrics
    from Semantic_Segmentation import Metrics
    f1 = Metrics.f1score(y_true,y_pred,average,weights)
    ''' this returns the f1 score , the descriptions for parameters will be shown once you tyoe the class name'''
    
 ## For converting data
    # To onehots
        cd = Semantic_Segmentation.convert_data(n_classes,data_y,color_map)
        data = cd.to_onehot
    
    # To segmented labels(RGB values)
        data = cd.to_segmentation(predicted_data)    
        
           
           
# Note  
     The current version cannot be used for binary class, only multi-class
  
  
# Requirements

This project is dependent on - Keras, Tensorflow, Numpy. If you dont meet the requirements execute the following commands,

For tensorflow --> pip install --upgrade tensorflow

For Keras      --> pip install --upgrade keras

For numpy      --> pip install --upgrade numpy
