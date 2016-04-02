from mnist import MNIST
import numpy as np
import random
from datetime import datetime
from skimage.feature import hog
class Data:
    '''
    This class loads the data and distributes it randomly into test/train sets
    '''
    def __init__(self,pixels_per_cell = (8,8),cells_per_block = (3,3),orientations=9):
        self.learning_set = []
        self.learning_set_labels = []
        self.load(pixels_per_cell,cells_per_block,orientations)

    def load(self,pixels_per_cell = (8,8),cells_per_block=(3,3),orientations=9):
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

        learning_set = []
        Boom = {}
        for i in range(10):
            Boom[str(i)] = []
        for i in range(0,60000):
            Boom[str(train_raw[1][i])].append(train_raw[0][i])
        for i in range(0,10000):
            Boom[str(test_raw[1][i])].append(test_raw[0][i])
        t = datetime.now().microsecond
        random.seed(t)
        [random.shuffle(Boom[str(i)]) for i in range(10)]

        print "Choosing 20000 training images uniformly randomly"

        # Descriptor Generator
        for l in range(10):
            for i in range(0,2000):
                img =  np.array(Boom[str(l)][i])
                img.shape = (28,28)
                fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block, visualise=True)
                learning_set.append([fd,l])

        print "Data Points now chosen and Generated HOG descriptors for them"

        t = datetime.now().microsecond
        random.seed(t)
        print "Shuffling Chosen Data Set"
        random.shuffle(learning_set)

        for i in range(20000):
            self.learning_set.append(learning_set[i][0])
            self.learning_set_labels.append(learning_set[i][1])

        print "Data Loading and Distribution Succesfully done"

        self.train_set = self.learning_set[:10000]
        self.train_labels = self.learning_set_labels[:10000]
        self.test_set = self.learning_set[10000:20000]
        self.test_labels = self.learning_set_labels[10000:20000]
