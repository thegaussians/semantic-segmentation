from keras.models import Model
from keras.layers import Add
from keras.regularizers import l2
from keras.applications import *
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Activation


class FCN32:

    """ FCN32 architecture, it doesn't use any skip connections and upsamples the image by the scale of 32.

        Usage :
            FCN32 = Semantic_Segmentation.models.FCN32(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
            model = FCN32.build()

        # Arguments
            input_shape : size of the input image ,in tuple.
            n_classes : the number of target class.
                        dtype --> int
            base_model : name of the pre-trained cnn model on top of which FCN is built.
                        dtype --> string  default --> 'vgg16'
            weight_path : path to the pre_trained weight files.
                        dtype --> string  default --> 'imagenet'
            regularizer : the regularizing value, it uses L2 regularizers on the kernel/filters.
                        dtype --> float   default -->None
    """
    weight_decay = lambda self,x: l2(x) if type(x)==int else None


    def __init__(self,input_shape, n_classes,base_model='vgg16',weight_path='imagenet',regularizer = None):
        self.base_model = base_model
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.weightpath = weight_path
        self.regularizer = regularizer


    def encoder(self):

        """ Builds the encoder layer on top of the given pre-trained CNN model

            # Returns
                model_en : the architecture of the encoder part
        """
        if self.base_model == 'vgg16':
            base = vgg16.VGG16(include_top=False, weights=self.weightpath, pooling=None, input_shape=self.input_shape)
        elif self.base_model == 'vgg19':
            base = vgg19.VGG19(include_top=False, weights=self.weightpath, pooling=None, input_shape=self.input_shape)
        else: base = resnet50.ResNet50(include_top=False, weights=self.weightpath, pooling=None, input_shape=self.input_shape)

        layer_encoder = Conv2D(4096, (7, 7), padding='same', activation='relu', kernel_regularizer=self.weight_decay(self.regularizer),
                               name='conv_en1')(base.output)
        layer_encoder = Conv2D(4096, (1, 1), padding='same', activation='relu', kernel_regularizer=self.weight_decay(self.regularizer),
                               name='conv_en2')(layer_encoder)

        encoder_model = Model(base.input,layer_encoder)
        return encoder_model


    def __decoder(self,model_en):

        """ Builds the decoder layer on top of the encoder part.

            # Arguments
                model_en : the model architecture of the encoder part

            # Returns
                model : the full model architecture, both encoder and decoder combined
        """
        layer_decoder = Conv2D(self.n_classes, (1,1), padding='same', activation='relu', kernel_regularizer=self.weight_decay(self.regularizer),
                               name = 'conv_dec')(model_en.output)
        layer_decoder = Conv2DTranspose(self.n_classes, (64,64), strides=(32,32), padding = 'same', kernel_regularizer = self.weight_decay(self.regularizer),
                                        name = 'deconv')(layer_decoder)
        output = Activation('softmax',name='softmax')(layer_decoder)

        model = Model(model_en.input,output,name='FCN32-'+self.base_model)
        return model


    def skip_connections(self,model_en):

        """ Builds the skip connections based on the pre-trained CNN models for FCN16 and FCN8.
            NOTE- FCN32 doesnt use this function, its a base for other class/FCN16,FCN8

            # Arguments:
                model_en : the model architecture of the encoder part
        """
        if self.base_model == 'resnet50': skip_1,skip_2 = model_en.get_layer(model_en.layers[112].name).output, model_en.get_layer(model_en.layers[50].name).output
        else: skip_1,skip_2 = model_en.get_layer('block4_pool').output, model_en.get_layer('block3_pool').output
        return skip_1,skip_2



    def build(self):

        """ Invokes the building process

            # Returns
                model : the compelete model architecture, both encoder and decoder combined
        """
        encoder_model = self.encoder()
        model = self.__decoder(encoder_model)
        model.summary()
        return model



class FCN16(FCN32):

    """ FCN16 architecture, it uses one skip connection and upsamples the image by the scale of 16.

        Usage :
            FCN32 = Semantic_Segmentation.models.FCN16(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
            model = FCN16.build()

        # Arguments
            input_shape : size of the input image ,in tuple.
            n_classes : the number of target class.
                        dtype --> int
            base_model : name of the pre-trained cnn model on top of which FCN is built.
                        dtype --> string  default --> 'vgg16'
            weight_path : path to the pre_trained weight files.
                        dtype --> string  default --> 'imagenet'
            regularizer : the regularizing value, it uses L2 regularizers on the kernel/filters.
                        dtype --> float   default -->None
    """

    ## NOTE : It wraps the properties of FCN32, same arguments and encoder part but it uses diff decoder function
    ##since the decoder includes skip connection.
    def __decoder(self,model_en):

        skip,_ = super().skip_connections(model_en)

        layer_decoder = Conv2DTranspose(skip.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                        kernel_regularizer=super().weight_decay(self.regularizer), name='deconv1')(model_en.output)
        layer_decoder = Add(name='skip1')([layer_decoder, skip])
        layer_decoder = Conv2DTranspose(self.n_classes, (32,32), strides=(16, 16), padding='same',
                                        kernel_regularizer=super().weight_decay(self.regularizer), name='deconv3')(layer_decoder)
        output = Activation('softmax',name='softmax')(layer_decoder)

        model = Model(model_en.input, output,name='FCN16-'+self.base_model)
        return model


    def build(self):

        """ Invokes the building process

            # Returns
                model : the compelete model architecture, both encoder and decoder combined
        """
        encoder_model = super().encoder()
        model = self.__decoder(encoder_model)
        model.summary()
        return model


class FCN8(FCN32):

    """ FCN8 architecture, it uses two skip connections and upsamples the image by the scale of 8, preserves most of the information.

        Usage :
            FCN32 = Semantic_Segmentation.models.FCN16(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
            model = FCN16.build()

        # Arguments
            input_shape : size of the input image ,in tuple.
            n_classes : the number of target class.
                        dtype --> int
            base_model : name of the pre-trained cnn model on top of which FCN is built.
                        dtype --> string  default --> 'vgg16'
            weight_path : path to the pre_trained weight files.
                        dtype --> string  default --> 'imagenet'
            regularizer : the regularizing value, it uses L2 regularizers on the kernel/filters.
                        dtype --> float   default -->None
    """

    ## NOTE : It wraps the properties of FCN32, same arguments and encoder part but it uses diff decoder function
    ##since the decoder includes 2 skip connections.

    def __decoder(self,model_en):

        skip_1,skip_2 = super().skip_connections(model_en)

        layer_decoder = Conv2DTranspose(skip_1.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                        kernel_regularizer=super().weight_decay(self.regularizer), name='deconv1')(model_en.output)
        layer_decoder = Add(name='skip1')([layer_decoder, skip_1])
        layer_decoder = Conv2DTranspose(skip_2.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                        kernel_regularizer=super().weight_decay(self.regularizer), name='deconv2')(layer_decoder)
        layer_decoder = Add(name='skip2')([layer_decoder, skip_2])
        layer_decoder = Conv2DTranspose(self.n_classes, (16, 16), strides=(8, 8), padding='same',
                                        kernel_regularizer=super().weight_decay(self.regularizer), name='deconv3')(layer_decoder)

        output = Activation('softmax',name='softmax')(layer_decoder)
        model = Model(model_en.input,output,name='FCN8-'+self.base_model)
        return model


    def build(self):

        """ Invokes the building process

            # Returns
                model : the compelete model architecture, both encoder and decoder combined
        """
        encoder_model = super().encoder()
        model = self.__decoder(encoder_model)
        model.summary()
        return model

