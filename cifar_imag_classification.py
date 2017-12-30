#for python 3

import numpy as np
import os
#from numbapro import vectorize

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
        Y_function = np.zeros((10000,10), dtype = np.uint8)
        Yraw = np.asarray(dict[b'labels'])
        for i in range(10000):
            Y_function[i,Yraw[i]] = 1
            
        #for getting file/image file names
        names_function = np.asarray(dict[b'filenames'])
        
        return X_function,Y_function,names_function

    
class NearestNeighbor(object):
      def __init__(self):
        pass
        
      def train(self, X, Y):
        """ X is (50000,3072) where each row is an example. Y is of size (50000,10) """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtrain = X
        self.Ytrain = Y

      #@vectorize(["uint8(uint8)"],target = 'gup')      trying uot on cuda (not tested)
      def predict(self, X):
        """ X is (10000,3072) where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        print(np.shape(num_test))
        # lets make an array Ypred which will store values of our prediction
        Ypred = np.zeros((num_test,10))
        print(np.shape(Ypred))
    
        # loop over all test rows
        for i in range(num_test):
          # find the nearest training image to the i'th test image
          # using the L1 distance (sum of absolute value differences)
          distances = np.sum(np.abs(self.Xtrain - X[i,:]), axis = 1)
          min_index = np.argmin(distances) # get the index with smallest distance
          Ypred[i] = self.Ytrain[min_index] # predict the label of the nearest example
          print(i)
    
        return Ypred



#training data batches
#X = shape(50000,3072), Y = shape(50000,10), names = shape(50000,)

X, Y, names = get_data('data_batch_1')          #testing first batch
for i in range(2,6):                            #adding remaining 4 batches
    X_temp, Y_temp, names_temp = get_data('data_batch_' + str(i))
    X = np.concatenate((X,X_temp),axis = 0)
    Y = np.concatenate((Y,Y_temp),axis = 0)
    names = np.concatenate((names,names_temp),axis = 0)

#test data batch 
#X_test = shape(10000,3072), Y_test = shape(10000,10), names_test = shape(10000,)
X_test,Y_test,names_test = get_data('test_batch')  

#twsting out Nearest Neighbour
nn = NearestNeighbor()
nn.train(X,Y)
prediction = nn.predict(X_test)

#printing classification accuracy
print ('accuracy: %f' % ( np.mean(prediction == Y_test) ))


