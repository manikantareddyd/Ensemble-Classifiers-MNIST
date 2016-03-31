from mnist import MNIST
import numpy as np
import random
from datetime import datetime
from skimage.feature import hog
class Data:
    '''
    This class loads the data and distributes it randomly into test/train sets
    '''
    def __init__(self,cell_size = (16,16)):
        self.train_set,self.train_labels,self.test_set,self.test_labels = self.load(cell_size)

    def load(self,cell_size = (16,16)):
        '''
        Generates a Data Set

        Parameters: None

        Returns:    train_set     - Training Set of 10000 images
                    train_labels  - Training Set Labels of corresponding images
                    test_set      - Test Set of 10000 images
                    test_labels   - Test Set Labels of corresponding images
        '''
        mn = MNIST('./data')
        train_raw = mn.load_training()
        test_raw = mn.load_testing()

        print "Loaded Raw images"
        train_labels = []
        train_data = []
        test_labels = []
        test_data=[]
        for i in range(0,60000):
        	train_labels.append(train_raw[1][i])
        	train_data.append(train_raw[0][i])
        for i in range(0,10000):
        	test_labels.append(test_raw[1][i])
        	test_data.append(test_raw[0][i])

        print "Choosing 10000 training images uniformly randomly"
        count=[]
        for i in range(0,10):
            count.append(1000)

        training_data = {}
        iter=0
        numbers = [i for i in range(60000)]
        t = datetime.now().microsecond
        random.seed(t)
        random.shuffle(numbers)
        for i in numbers:
            if count[int(train_labels[i])]>0:
                count[int(train_labels[i])]-=1
                training_data[iter]=(train_labels[i],train_data[i])
                iter+=1

        numbers = [i for i in range(len(training_data))]
        t = datetime.now().microsecond
        random.seed(t)
        random.shuffle(numbers)
        training_set=[]
        training_set_labels=[]
        # Descriptor Generator
        for i in numbers:
            img =   np.array(training_data[i][1])
            img.shape = (28,28)
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=cell_size,cells_per_block=(1, 1), visualise=True)
            training_set.append(fd)
            training_set_labels.append(training_data[i][0])

        test_data_set = []
        for i in range(len(test_data)):
            img = np.array(test_data[i])
            img.shape=(28,28)
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=cell_size, cells_per_block=(1, 1), visualise=True)
            test_data_set.append(fd)

        print "Data Loading and Distribution Succesfully done"
        return training_set,training_set_labels,test_data_set,test_labels
