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
        for i in range(0,60000):
            learning_set.append((train_raw[0][i],train_raw[1][i]))
        for i in range(0,10000):
            learning_set.append((test_raw[0][i],test_raw[1][i]))

        print "Choosing 20000 training images uniformly randomly"

        t = datetime.now().microsecond
        random.seed(t)
        random.shuffle(learning_set)
        print "Chosen"
        # Descriptor Generator
        for i in range(0,20000):
            img =   np.array(learning_set[i][0])
            img.shape = (28,28)
            fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block, visualise=True)
            self.learning_set.append(fd)
            self.learning_set_labels.append(learning_set[i][1])

        print "Data Loading and Distribution Succesfully done"
        # self.train_set = self.learning_set[:10000]
        # self.train_labels = self.learning_set_labels[:10000]
        # self.test_set = self.learning_set[10000:20000]
        # self.test_labels = self.learning_set_labels[10000:20000]
