from data import *
from math import *
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn import tree
from numpy.random import choice
import math
from sklearn.preprocessing import normalize

class DTC:
    def __init__(self,train_set,train_labels,depth,criterion='c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        self.clf  = tree.DecisionTreeClassifier(criterion=c,splitter='random',max_depth=depth)
        self.clf.fit(train_set,train_labels)

class AdaBoost:
    def __init__(self, Data, NoOfClassifiers = 50,L=5000,depth=10):
        self.depth = depth
        n=int(1*L)
        self.NoOfClassifiers = NoOfClassifiers
        Data.learning_set=Data.learning_set[:L]
        Data.learning_set_labels=Data.learning_set_labels[:L]
        print "Choosing ",L," of the Dataset"
        self.alpha = []
        self.models = []
        self.BoostedPredictions=[]
        self.Boost(Data,n)
        self.BoostedPredVec=[]
        self.getPredictions(Data)
        self.BoostedScore = 0
        for i in range(L):
            if str(self.BoostedPredictions[i]) == str(Data.learning_set_labels[i]):
                self.BoostedScore += 1.0/(L*1.0)
        print self.NoOfClassifiers,self.BoostedScore

    def loss(self,true_label,predicted_label):
        '''
        loss function
        Can be varied. Currently Binary
        '''
        if true_label != predicted_label:
            return 1
        else:
            return 0

    def Boost(self,Data,train_set_size):
        '''
        Boosting Algorithm...
        '''
        print "Boosting"
        L = len(Data.learning_set)
        p = [1.0/(1.0*L) for i in range(L)]
        for j in range(self.NoOfClassifiers):
            X_train, y_train = self.genTrainSet(Data,p,train_set_size)
            clf = DTC(X_train,y_train, depth = self.depth)
            predictions = clf.clf.predict(Data.learning_set)
            error = 1.0 - clf.clf.score(Data.learning_set,Data.learning_set_labels)
            if error > 0.5:
                self.NoOfClassifiers = j
                print "Not a Weak Classifier"
                print "Now no of classes:", self.NoOfClassifiers
                break
            else:
                self.models.append(clf)
                self.alpha.append(0.5*log((1-error)/(1.0*error)))
                # print "Error at ",j , error,'And its alpha ',self.alpha[j]
                for i in range(0,L):
                    if predictions[i] == Data.learning_set_labels[i]:
                        p[i] = p[i]*(1.0*sqrt(1.0/(1.0*self.alpha[j])))
                    else:
                        p[i] = p[i]*(1.0*sqrt(1.0*self.alpha[j]))
                z = sum(p)
                for i in range(len(p)):
                    p[i]=(1.0*p[i])/(1.0*z)

    def genTrainSet(self,Data, weights, train_set_size):
        '''
        Chooses a subset of the training set of size train_set_size
        with a probability distribution~weights
        '''
        indices = [i for i in range(len(Data.learning_set))]
        indexlist = [choice(indices, p=weights) for i in range(train_set_size)]
        X_train = []
        y_train = []
        for i in indexlist:
            X_train.append(Data.learning_set[i])
            y_train.append(Data.learning_set_labels[i])
        return X_train,y_train

    def getPredictions(self,Data):
        '''
        Generates Predictions based on the Boosted Model
        '''
        for i in range(len(Data.learning_set)):
            # print self.NoOfClassifiers
            probvec=np.array([0 for t in range(10)],dtype='float64')
            for j in range(self.NoOfClassifiers):
                vec = (self.models[j].clf.predict_proba([Data.learning_set[i]]))[0]
                # print vec.shape,self.alpha[j]
                vec = np.array(vec,dtype='float64')
                # print vec.shape
                probvec += (self.alpha[j])*vec
            norm = np.linalg.norm(probvec)
            poo = [probvec[t]*1.0/norm for t in range(10)]
            self.BoostedPredVec.append(poo)
            self.BoostedPredictions.append(np.argmax(poo))

# D=Data()
# f=open('DataWeakClassifiers.pkl','wb')
# import pickle
# pickle.dump(D,f,pickle.HIGHEST_PROTOCOL)
# f.close()


import pickle
f=open('DataWeakClassifiers.pkl','rb')
D = pickle.load(f)
f.close()
e=AdaBoost(D)
print e.BoostedScore
