from keras.backend import tensorflow_backend as K
import tensorflow as tf
from .Losses import focal_loss



class Metrics:

    __weighted_method = lambda self,x,y,string,w: (K.sum(x)/K.sum(y)) if string=='inter' else (K.sum(w*x)/K.sum(w*y))
    __avg_method = lambda self,x,y,string,w: K.mean(x/y) if string=='intra' else self.__weighted_method(x,y,string,w)


    def __metrics_base(self,y_true,y_pred):

        """ Base for all the metrics defined below """
        y_true,y_pred = K.flatten(tf.math.argmax(y_true,axis=-1)), K.flatten(tf.math.argmax(y_pred,axis=-1))
        con_mat = K.cast(tf.math.confusion_matrix(y_true,y_pred), K.floatx())
        correct = tf.linalg.diag_part(con_mat)
        total = K.sum(con_mat, axis=-1)
        return correct,total,con_mat


    def accuracy(self,y_true,y_pred):

        """ computes the accuracy

            # Arguments
                y_true : target value
                y_pred : predicted class value
            # Returns
                acc : overall accuracy
        """
        correct,total,_ = self.__metrics_base(y_true,y_pred)
        return ( K.sum(correct) / K.sum(total) )


    def IoU(self, y_true, y_pred,average='inter',weights=None):

        """ Intersection over Union , IoU = A^B/(A U B - A^B)
           Computes the percentage overlap with the target image.

            # Arguments
                y_true : target value
                y_pred : predicted class value
                average : 'inter' --> computes the IoU score overall  'intra' --> computes the score for each calss and computes the average
                        'weighted' --> computes the weighted average , useful for imabalanced class.
                weights :  only if average is specified 'weighted', weights for the respective classes.
            # Returns
                IoU score
        """
        _, _, con_mat = self.__metrics_base(y_true, y_pred)
        intersection = tf.linalg.diag_part(con_mat)
        ground_truth_set = K.sum(con_mat, axis=1)
        predicted_set = K.sum(con_mat, axis=0)
        union = ground_truth_set + predicted_set - intersection
        return self.__avg_method(intersection,union,average,weights)


    def recall(self,y_true,y_pred,average='inter',weights=None):

        """ Computes the recall score over each given class and gives the overall score.  recall = TP/TP+FN

            # Arguments
                y_true : target value
                y_pred : predicted class value
                average : 'inter' --> computes the recall score overall  'intra' --> computes the score for each calss and computes the average
                        'weighted' --> computes the weighted average , useful for imabalanced class.
                weights :  only if average is specified 'weighted', weights for the respective classes.
            # Returns
                recall score
        """
        correct,total,_ = self.__metrics_base(y_true,y_pred)
        return self.__avg_method(correct,total,average,weights)


    def precision(self,y_true,y_pred,average='inter',weights=None):

        """ Computes the precision over each given class and returns the overall score.  precision = TP/TP+FP

            # Arguments
                y_true : target value
                y_pred : predicted class value
                average : 'inter' --> computes the precision score overall  'intra' --> computes the score for each calss and computes the average
                        'weighted' --> computes the weighted average , useful for imabalanced class.
                weights :  only if average is specified 'weighted', weights for the respective classes.
            # Returns
                precision score
        """
        correct,_,con_mat = self.__metrics_base(y_true,y_pred)
        total = K.sum(con_mat,axis=0)
        return self.__avg_method(correct,total,average,weights)


    def f1score(self,y_true,y_pred,average='inter',weights=None):

        """ Computes the f1 score over each given class and returns the overall score.  f1 = (2*precision*recall)/(precision+recall)

            # Arguments
                y_true : target value
                y_pred : predicted class value
                average : 'inter' --> computes the f1 score overall  'intra' --> computes the score for each calss and computes the average
                            'weighted' --> computes the weighted average , useful for imabalanced class.
                weights :  only if average is specified 'weighted', weights for the respective classes.
            # Returns
                 f1 score
        """
        precision = self.precision(y_true,y_pred,average,weights)
        recall = self.recall(y_true,y_pred,average,weights)
        return ((2*precision*recall)/(precision+recall))



    def dice_coeffiecient(self,y_true,y_pred,average='inter',weights=None):
        """ Computes the dice score over each given class and returns the overall score.

                # Arguments
                    y_true : target value
                    y_pred : predicted class value
                    average : 'inter' --> computes the dice score overall  'intra' --> computes the score for each calss and computes the average
                                    'weighted' --> computes the weighted average , useful for imabalanced class.
                    weights :  only if average is specified 'weighted', weights for the respective classes.
                # Returns
                    dice score
                """

        y_pred = focal_loss.clipping(y_pred)
        intersection = 2 * K.sum((y_true * y_pred),axis=(0,1,2))
        union = K.sum( (y_true*y_true) + (y_pred*y_pred),axis=(0,1,2))
        return self.__avg_method(intersection,union,average,weights)
