from keras.models import Model
from keras.layers import Add, Input, MaxPooling2D, concatenate, BatchNormalization
from keras.regularizers import l2
from keras.applications import *
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation



class UNET:


    class ResUnet:

        """ Combining the power of residual network with Unet to achieve state of the art performance
            'Deep residual Unet'

            Arguments:
                input_shape - shape of the input image,   dtype: tuple
                n_classes - number of target classes,   dtype: int
                summary - If True prints the model summary and vice-versa,  dtype: bool   default: True

            Returns:
                model - The model architecture of ResUnet
        """

        def __init__(self, input_shape, n_classes, summary=True):

            self.input_shape = input_shape
            self.n_classes = n_classes
            self.summary = summary


        def __bn_act(self, x, act=True):

            x = BatchNormalization()(x)
            if act == True:
                x = Activation("relu")(x)
            return x


        def __conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):

            conv = self.__bn_act(x)
            conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
            return conv


        def __stem(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):

            conv = Conv2D(filters, kernel_size, padding=padding, strides=strides,activation='relu')(x)
            conv = self.__conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

            shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
            shortcut = self.__bn_act(shortcut, act=False)

            output = Add()([conv, shortcut])
            return output


        def __residual_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):

            res = self.__conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
            res = self.__conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

            shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
            shortcut = self.__bn_act(shortcut, act=False)

            output = Add()([shortcut, res])
            return output


        def __upsample_concat_block(self, x,filters, xskip):

            u = Conv2DTranspose(strides=(2, 2),kernel_size=4,padding='same',filters=filters)(x)
            c = concatenate([u, xskip])
            return c


        def build(self):

            f = [16, 32, 64, 128, 256]
            inputs = Input(self.input_shape)

            ## Encoder
            e0 = inputs
            e1 = self.__stem(e0, f[0])
            e2 = self.__residual_block(e1, f[1], strides=2)
            e3 = self.__residual_block(e2, f[2], strides=2)
            e4 = self.__residual_block(e3, f[3], strides=2)
            e5 = self.__residual_block(e4, f[4], strides=2)

            ## Bridge
            b0 = self.__conv_block(e5, f[4], strides=1)
            b1 = self.__conv_block(b0, f[4], strides=1)

            ## Decoder
            u1 = self.__upsample_concat_block(b1,f[4], e4)
            d1 = self.__residual_block(u1, f[4])

            u2 = self.__upsample_concat_block(d1,f[3], e3)
            d2 = self.__residual_block(u2, f[3])

            u3 = self.__upsample_concat_block(d2,f[2], e2)
            d3 = self.__residual_block(u3, f[2])

            u4 = self.__upsample_concat_block(d3,f[1], e1)
            d4 = self.__residual_block(u4, f[1])

            outputs = Conv2D(self.n_classes, (1, 1), padding="same", activation="relu")(d4)
            outputs = Activation('softmax')(outputs)

            model = Model(inputs, outputs)

            if self.summary: print(model.summary())
            return model



    class VanilaUnet:

        """ Unet architecture

               Usage :
                   unet = Semantic_Segmentation.models.UNET.unet(input_shape,no_classes,regularizer,summary)
                   model = unet.build()

               # Arguments
                   input_shape : size of the input image ,in tuple.
                   n_classes : the number of target class.
                               dtype --> int
                   regularizer : the regularizing value, it uses L2 regularizers on the kernel/filters.
                               dtype --> float   default -->None
                   Batchnorm : If true performs batchnorm over each block
                                dtype --> bool   default --> True
                   summary : If True prints the model summary
                                default --> True
        """
        build = lambda self: self.__architecture()
        weight_decay = lambda self, x: None if x == None else l2(x)
        batchnorm = lambda self, x: BatchNormalization(beta_regularizer=l2(0.001),
                                                       gamma_regularizer=l2(0.001)) if x else Activation('linear')

        def __init__(self, input_shape, n_classes, regularizer=None, BatchNorm=True, summary=True):
            self.input_shape = input_shape
            self.n_classes = n_classes
            self.regularizer = regularizer
            self.BatchNorm = BatchNorm
            self.summary = summary

        def __architecture(self):
            input = Input(self.input_shape)

            conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(input)
            batchnorm1 = self.batchnorm(self.BatchNorm)(conv1)
            activation1 = Activation('relu')(batchnorm1)

            conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation1)
            batchnorm1 = self.batchnorm(self.BatchNorm)(conv1)
            activation1 = Activation('relu')(batchnorm1)

            pool1 = MaxPooling2D(pool_size=(2, 2))(activation1)


            conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(pool1)
            batchnorm2 = self.batchnorm(self.BatchNorm)(conv2)
            activation2 = Activation('relu')(batchnorm2)

            conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation2)
            batchnorm2 = self.batchnorm(self.BatchNorm)(conv2)
            activation2 = Activation('relu')(batchnorm2)

            pool2 = MaxPooling2D(pool_size=(2, 2))(activation2)


            conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(pool2)
            batchnorm3 = self.batchnorm(self.BatchNorm)(conv3)
            activation3 = Activation('relu')(batchnorm3)

            conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation3)
            batchnorm3 = self.batchnorm(self.BatchNorm)(conv3)
            activation3 = Activation('relu')(batchnorm3)

            pool3 = MaxPooling2D(pool_size=(2, 2))(activation3)


            conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(pool3)
            batchnorm4 = self.batchnorm(self.BatchNorm)(conv4)
            activation4 = Activation('relu')(batchnorm4)

            conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation4)
            batchnorm4 = self.batchnorm(self.BatchNorm)(conv4)
            activation4 = Activation('relu')(batchnorm4)

            pool4 = MaxPooling2D(pool_size=(2, 2))(activation4)


            conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(pool4)
            batchnorm5 = self.batchnorm(self.BatchNorm)(conv5)
            activation5 = Activation('relu')(batchnorm5)
            
            conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation5)
            batchnorm5 = self.batchnorm(self.BatchNorm)(conv5)
            activation5 = Activation('relu')(batchnorm5)

            up5 = Conv2DTranspose(512, 4, strides=(2, 2), padding='same',
                                  kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_5')(activation5)


            merge6 = concatenate([activation4, up5], axis=3)
            conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(merge6)
            batchnorm6 = self.batchnorm(self.BatchNorm)(conv6)
            activation6 = Activation('relu')(batchnorm6)

            conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation6)
            batchnorm6 = self.batchnorm(self.BatchNorm)(conv6)
            activation6 = Activation('relu')(batchnorm6)

            up6 = Conv2DTranspose(256, 4, strides=(2, 2), padding='same',
                                  kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_6')(activation6)


            merge7 = concatenate([activation3, up6], axis=3)
            conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(merge7)
            batchnorm7 = self.batchnorm(self.BatchNorm)(conv7)
            activation7 = Activation('relu')(batchnorm7)

            conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation7)
            batchnorm7 = self.batchnorm(self.BatchNorm)(conv7)
            activation7 = Activation('relu')(batchnorm7)

            up7 = Conv2DTranspose(128, 4, strides=(2, 2), padding='same',
                                  kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_7')(activation7)


            merge8 = concatenate([activation2, up7], axis=3)
            conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
            batchnorm8 = self.batchnorm(self.BatchNorm)(conv8)
            activation8 = Activation('relu')(batchnorm8)

            conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(activation8)
            batchnorm8 = self.batchnorm(self.BatchNorm)(conv8)
            activation8 = Activation('relu')(batchnorm8)

            up8 = Conv2DTranspose(64, 4, strides=(2, 2), padding='same',
                                  kernel_regularizer=self.weight_decay(self.regularizer), name='upsample_8')(activation8)


            merge9 = concatenate([activation1, up8], axis=3)
            conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(merge9)
            batchnorm9 = self.batchnorm(self.BatchNorm)(conv9)
            activation9 = Activation('relu')(batchnorm9)

            conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation9)
            batchnorm9 = self.batchnorm(self.BatchNorm)(conv9)
            activation9 = Activation('relu')(batchnorm9)

            conv9 = Conv2D(self.n_classes, 1, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=self.weight_decay(self.regularizer))(activation9)
            op = Activation('softmax', name='softmax')(conv9)

            model = Model(input=input, output=op, name='Unet')
            if self.summary: print(model.summary())
            return model


class FCN:
    """ Available models --> fcn32, fcn16, fcn8

        All the three avilable with vgg16,vgg19,resnet50
    """

    class fcn32:

        """ FCN32 architecture, it doesn't use any skip connections and upsamples the image by the scale of 32.

            Usage :
                FCN32 = Semantic_Segmentation.models.FCN.fcn32(input_shape,no_classes,base_model_name,pretrained_weights,regularizer,summary)
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
                summary : If True prints the model summary
                                default --> True
        """
        weight_decay = lambda self, x: l2(x) if type(x) == int else None
        build = lambda self: self.__decoder(self.encoder())

        def __init__(self, input_shape, n_classes, base_model='vgg16', weight_path='imagenet', regularizer=None,
                     summary=True):
            self.base_model = base_model
            self.input_shape = input_shape
            self.n_classes = n_classes
            self.weightpath = weight_path
            self.regularizer = regularizer
            self.summary = summary

        def encoder(self):

            """ Builds the encoder layer on top of the given pre-trained CNN model

                # Returns
                    model_en : the architecture of the encoder part
            """
            if self.base_model == 'vgg16':
                base = vgg16.VGG16(include_top=False, weights=self.weightpath, pooling=None,
                                   input_shape=self.input_shape)
            elif self.base_model == 'vgg19':
                base = vgg19.VGG19(include_top=False, weights=self.weightpath, pooling=None,
                                   input_shape=self.input_shape)
            else:
                base = resnet50.ResNet50(include_top=False, weights=self.weightpath, pooling=None,
                                         input_shape=self.input_shape)

            layer_encoder = Conv2D(4096, (7, 7), padding='same', activation='relu',
                                   kernel_regularizer=self.weight_decay(self.regularizer),
                                   name='conv_en1')(base.output)
            layer_encoder = Conv2D(4096, (1, 1), padding='same', activation='relu',
                                   kernel_regularizer=self.weight_decay(self.regularizer),
                                   name='conv_en2')(layer_encoder)

            encoder_model = Model(base.input, layer_encoder)
            return encoder_model

        def __decoder(self, model_en):

            """ Builds the decoder layer on top of the encoder part.

                # Arguments
                    model_en : the model architecture of the encoder part

                # Returns
                    model : the full model architecture, both encoder and decoder combined
            """
            layer_decoder = Conv2D(self.n_classes, (1, 1), padding='same', activation='relu',
                                   kernel_regularizer=self.weight_decay(self.regularizer),
                                   name='conv_dec')(model_en.output)
            layer_decoder = Conv2DTranspose(self.n_classes, (64, 64), strides=(32, 32), padding='same',
                                            kernel_regularizer=self.weight_decay(self.regularizer),
                                            name='deconv')(layer_decoder)
            output = Activation('softmax', name='softmax')(layer_decoder)

            model = Model(model_en.input, output, name='FCN32-' + self.base_model)
            if self.summary: print(model.summary())
            return model

        def skip_connections(self, model_en):

            """ Builds the skip connections based on the pre-trained CNN models for FCN16 and FCN8.
                NOTE- FCN32 doesnt use this function, its a base for other class/FCN16,FCN8

                # Arguments:
                    model_en : the model architecture of the encoder part
            """
            if self.base_model == 'resnet50':
                skip_1, skip_2 = model_en.get_layer(model_en.layers[112].name).output, model_en.get_layer(
                    model_en.layers[50].name).output
            else:
                skip_1, skip_2 = model_en.get_layer('block4_pool').output, model_en.get_layer('block3_pool').output
            return skip_1, skip_2

    class fcn16(fcn32):

        """ FCN16 architecture, it uses one skip connection and upsamples the image by the scale of 16.

            Usage :
                FCN16 = Semantic_Segmentation.models.FCN.fcn16(input_shape,no_classes,base_model_name,pretrained_weights,regularizer,summary)
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
                summary : If True prints the model summary
                                default --> True
        """

        ## NOTE : It wraps the properties of FCN32, same arguments and encoder part but it uses diff decoder function
        ##since the decoder includes skip connection.
        build = lambda self: self.__decoder(super().encoder())

        def __decoder(self, model_en):
            skip, _ = super().skip_connections(model_en)

            layer_decoder = Conv2DTranspose(skip.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                            kernel_regularizer=super().weight_decay(self.regularizer), name='deconv1')(
                model_en.output)
            layer_decoder = Add(name='skip1')([layer_decoder, skip])
            layer_decoder = Conv2DTranspose(self.n_classes, (32, 32), strides=(16, 16), padding='same',
                                            kernel_regularizer=super().weight_decay(self.regularizer), name='deconv3')(
                layer_decoder)
            output = Activation('softmax', name='softmax')(layer_decoder)

            model = Model(model_en.input, output, name='FCN16-' + self.base_model)
            if self.summary: print(model.summary())
            return model

    class fcn8(fcn32):

        """ FCN8 architecture, it uses two skip connections and upsamples the image by the scale of 8, preserves most of the information.

            Usage :
                FCN8 = Semantic_Segmentation.models.FCN.fcn8(input_shape,no_classes,base_model_name,pretrained_weights,regularizer,summary)
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
                summary : If True prints the model summary
                                default --> True
        """

        ## NOTE : It wraps the properties of FCN32, same arguments and encoder part but it uses diff decoder function
        ##since the decoder includes 2 skip connections.

        build = lambda self: self.__decoder(super().encoder())

        def __decoder(self, model_en):
            skip_1, skip_2 = super().skip_connections(model_en)

            layer_decoder = Conv2DTranspose(skip_1.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                            kernel_regularizer=super().weight_decay(self.regularizer), name='deconv1')(
                model_en.output)
            layer_decoder = Add(name='skip1')([layer_decoder, skip_1])
            layer_decoder = Conv2DTranspose(skip_2.get_shape().as_list()[-1], (4, 4), strides=(2, 2), padding='same',
                                            kernel_regularizer=super().weight_decay(self.regularizer), name='deconv2')(
                layer_decoder)
            layer_decoder = Add(name='skip2')([layer_decoder, skip_2])
            layer_decoder = Conv2DTranspose(self.n_classes, (16, 16), strides=(8, 8), padding='same',
                                            kernel_regularizer=super().weight_decay(self.regularizer), name='deconv3')(
                layer_decoder)
            output = Activation('softmax', name='softmax')(layer_decoder)

            model = Model(model_en.input, output, name='FCN8-' + self.base_model)
            if self.summary: print(model.summary())
            return model




