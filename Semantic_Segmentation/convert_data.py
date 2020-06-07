import numpy as np

class convert_data:

    """ This deals with converting the segmented image to one-hots and converting back the one-hots to
    segmented image.

        Arguments :
            n_classes : the number of target classes
                        dtype --> int
            data_y : the target data
                        dtype --> array4D
            color_map : the RGB value info for the corredponding segmented class
                        dtype --> array2D

        For converting to one-hots:
            cd = convert_data(n_classes,data_y,color_map)
            data = cd.to_onehot

        For converting to segmented form:
            cd = convert_data(n_classes,data_y,color_map)
            data = cd.to_segmentation

    """

    def __init__(self,n_classes,data_y,color_map):

        self.n_classes = n_classes
        self.data_y = data_y
        self.color_map = color_map
        self.onehots = np.identity(self.n_classes)
        self.to_onehot = self.__convert(self.data_y, self.color_map, self.onehots, np.zeros((self.data_y.shape[0],self.data_y.shape[1],self.data_y.shape[2],self.n_classes)))

    def __convert(self,data1,data2,data3,data4):

        for i in range(self.n_classes): data4[(np.where((np.sum(abs(data1 - data2[i]), axis=-1)) == 0))] = data3[i]
        return data4

    def to_segmentation(self,predicted_data):

        return self.__convert(predicted_data,self.onehots,self.color_map,np.zeros((self.data_y.shape[0],self.data_y.shape[1],self.data_y.shape[2],3)))