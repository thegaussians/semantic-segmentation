from keras.models import Model
from keras.layers import Add,Input,MaxPooling2D,concatenate
from keras.regularizers import l2
from keras.applications import *
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Activation




class UNET:

    class unet:
        """ Unet architecture

               Usage :
                   unet = Semantic_Segmentation.models.UNET.unet(input_shape,no_classes,regularizer)
                   model = unet.build()

               # Arguments
                   input_shape : size of the input image ,in tuple.
                   n_classes : the number of target class.
                               dtype --> int
                   regularizer : the regularizing value, it uses L2 regularizers on the kernel/filters.
                               dtype --> float   default -->None
           """

        ## use build to execute the architecture building process,it returns the required model
        
        #build = lambda self: self.model
        weight_decay = lambda self, x: None if x == None else l2(x)

        def __init__(self,input_shape,n_classes,regularizer=None):
            self.input_shape = input_shape
            self.n_classes = n_classes
            self.regularizer = regularizer
            self.model = self.__architecture()


        def __architecture(self):

            input = Input(self.input_shape)

            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(input)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv5)
            up5 = Conv2DTranspose(512, 4, strides=(2,2), padding='same', kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_5')(conv5)

            merge6 = concatenate([conv4, up5], axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=self.weight_decay(self.regularizer))(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=self.weight_decay(self.regularizer))(conv6)
            up6 = Conv2DTranspose(256, 4, strides=(2,2), padding='same', kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_6')(conv6)

            merge7 = concatenate([conv3, up6], axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=self.weight_decay(self.regularizer))(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv7)
            up7 = Conv2DTranspose(128, 4, strides=(2,2), padding='same', kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_7')(conv7)

            merge8 = concatenate([conv2, up7], axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
            up8 = Conv2DTranspose(64, 4, strides=(2,2), padding='same', kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_8')(conv8)

            merge9 = concatenate([conv1, up8], axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv9)
            conv9 = Conv2D(self.n_classes, 1, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=self.weight_decay(self.regularizer))(conv9)
            op = Activation('softmax',name='softmax')(conv9)

            model = Model(input=input, output=op,name='Unet')
            return model


        def build(self):

            """ Invokes the model building process """
            self.model.summary()
            return self.model




class FCN:
    """ Available models --> fcn32, fcn16, fcn8

        All the three avilable with vgg16,vgg19,resnet50
    """

    class fcn32:

        """ FCN32 architecture, it doesn't use any skip connections and upsamples the image by the scale of 32.

            Usage :
                FCN32 = Semantic_Segmentation.models.FCN.fcn32(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
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



    class fcn16(fcn32):

        """ FCN16 architecture, it uses one skip connection and upsamples the image by the scale of 16.

            Usage :
                FCN16 = Semantic_Segmentation.models.FCN.fcn16(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
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


    class fcn8(fcn32):

        """ FCN8 architecture, it uses two skip connections and upsamples the image by the scale of 8, preserves most of the information.

            Usage :
                FCN8 = Semantic_Segmentation.models.FCN.fcn8(input_shape,no_classes,base_model_name,pretrained_weights,regularizer)
                model = FCN8.build()

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

