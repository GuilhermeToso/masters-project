""" 
Author: Guilherme Marino Toso
Title: preprocess
Project: Semi-Supervised Learning Using Competition for Neurons' Synchronization
Package: nsc

Description:
    The preprocess module contains the Preprocessing class 

"""



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessing():

    """ 
    .. py::class:

    This class allows the preprocessing of the data
    
    
    """

    def __init__(self, **kwargs):
        super().__init__()


    def split_data(self, data, target):

        """ 
        .. py::function::

        This method split the dataset into the features, numerical and categorical features and labels data sets.
        
        :param data: The DataFrame to be used as the data.
        :type data: pandas.core.frame.DataFrame
        :param target: The column name of the column target for classification.
        :type target: str 

        :return: The features, categoric, numeric and output dataframes
        :rtype: pandas.core.frame.DataFrame

        """

        if not isinstance(target,str):

            raise TypeError("{} must be of string type".format(target))

        if isinstance(data, pd.core.frame.DataFrame):

            features = data.drop(columns=[target], axis=1)
            categoric = features.select_dtypes(include=['object']).values
            numeric = features.select_dtypes(include=np.number).values
            output = data[target].values

            return features, categoric, numeric, output
        
        else:

            raise TypeError("Data must be of type pandas.core.frame.DataFrame")


    def label_encoder(self, categoric_data):

        """ 
        .. py::function::

        This method encode the labels of the categorical data. 
        
        :param categorical_data: The categorical data matrix.
        :type categorical_data: pandas.core.frame.DataFrame

        :return: A numpy representation of encoded categorical data
        :rtype: numpy.ndarray
        
        """

        if isinstance(categoric_data, pd.core.frame.DataFrame):

            if all(categoric_data.dtypes == 'object'):
            
                le = LabelEncoder()

                categoric_encoded = categoric_data.apply(le.fit_transform)
            
                return categoric_encoded

            else:

                raise TypeError("The categorical data columns must all have data of 'object' type")

        else:

            raise TypeError("The Categorical Data must be of type pandas.core.frame.DataFrame")

    
    def get_labels(self, data, target):

        """ 
        .. py::function::

        This method get the list of unique labels

        :param data: The DataFrame to be used as the data.
        :type data: pandas.core.frame.DataFrame
        :param target: The column name of the column target for classification.
        :type target: str 

        :return: A list of unique labels for the classification
        :rtype: list
        
        """

        if not isinstance(target,str):

            raise TypeError("{} must be of string type".format(target))

        if isinstance(data, pd.core.frame.DataFrame):

            labels = data[target].unique()
            labels = list(labels[labels != None])

            return labels
        
        else:

            raise TypeError("Data must be of type pandas.core.frame.DataFrame")


    def set_null_values(self, data, target, labels, **kwargs):

        """ 
        .. py::function::

        This method allows to set null values in a specific DataFrame column to represent
        unlabeled data points in that specific column (output column). You can either define 
        which data points you want to be unlabeled, or let the unlabel data be randomly choosed.
        
        :param data: The DataFrame to be used as the data.
        :type data: pandas.core.frame.DataFrame
        :param target: The column name of the column target for classification.
        :type target: str
        :param labels: A list of unique labels for the classification
        :type labels: list
        :param label_size (kwarg): The number of labeled data points of each label.
        :type label_size: int
        :param label_dict (kwarg): A dictionary which the *key:value* pair represents *label : list of indexes labeled by the key*.
        :type label_dict: dict
         

        :return: The data with null values in the output column.
        :rtype: pandas.core.frame.DataFrame

        """

        label_size = kwargs.get('label_size')

        label_dict = kwargs.get('label_dict')

        nulls_dict = {}

        index_array = np.array([])

        if label_size != None:

            if isinstance(label_size,int):
                
                for label in labels:
                
                    nulls_dict[label] = None
                
                    ind_array = data.loc[data[target] == label].index.to_numpy()
                
                    np.random.shuffle(ind_array)
    
                    ind_array = ind_array[label_size:]
                    
                    index_array = np.append(index_array,ind_array)

                data.loc[index_array, target] = data.loc[index_array, target].map(nulls_dict)

                return data

            else:

                raise TypeError("label_size must be an integer")

        elif label_dict != None:

            if isinstance(label_dict, dict):

                for label in label_dict.keys():
                    index_array = np.append(index_array,label_dict[label])
                    nulls_dict[label] = None

                inds = data.index.to_numpy()
                ind_array = np.setdiff1d(inds, index_array)
                data.loc[ind_array, target] = data.loc[ind_array, target].map(nulls_dict)
                return data
            
            else:

                raise TypeError("label_dict must be a dictionary")


    def get_label_unlabel_inds(self, data, target, labels):

        """ 
        .. py::function::

        This method get the indexes of the unlabeled data and a dictionary in which
        every key is a label and the value is a list of indexes of the data labeled with the respective key.

        :param data: The DataFrame to be used as the data.
        :type data: pandas.core.frame.DataFrame
        :param target: The column name of the column target for classification.
        :type target: str
        :param labels: A list of unique labels for the classification
        :type labels: list
        

        :return: A list of unlabel data indexes, and a dictionaire with key: value as label: list of label data indexes.
        :rtype: list, dict

        """
        if not (isinstance(data, pd.core.frame.DataFrame) and isinstance(target, str) and isinstance(labels,list)):
            raise TypeError("Some input argument has a wrong data type. 'Data' must be a pandas.DataFrame, \
                'target' must be a string and 'labels' must be a list.")


        unlabel_bool_df = pd.isnull(data).any(1)
        unlabel_indexes = unlabel_bool_df[unlabel_bool_df==True].index.to_numpy()


        labels_indexes = {}

        for label in labels:

            label_indexes = data.index[data[target] == label].to_numpy()
            labels_indexes[label] = label_indexes      

        return unlabel_indexes, labels_indexes

