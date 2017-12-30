#for python 3

import numpy as np
import os


#function mentioned in website
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(file):
        absFile = os.path.abspath(file)
        dict = unpickle(absFile)
        ''' to check keys........................
        for key in dict.keys():
            print(key)
        '''
        #for getting data values
        X_function = np.asarray(dict[b'data']).astype("uint8")
        
        #for getting lables
        Y_function = np.zeros((10000,10))
        Yraw = np.asarray(dict[b'labels'])
        for i in range(10000):
            Y_function[i,Yraw[i]] = 1
            
        #for getting file/image file names
        names_function = np.asarray(dict[b'filenames'])
        
        return X_function,Y_function,names_function



X, Y, names = get_data('data_batch_1')          #testing first batch
for i in range(2,6):                            #adding remaining 4 batches
    X_temp, Y_temp, names_temp = get_data('data_batch_' + str(i))
    X = np.concatenate((X,X_temp),axis = 0)
    Y = np.concatenate((Y,Y_temp),axis = 0)
    names = np.concatenate((names,names_temp),axis = 0)
    
