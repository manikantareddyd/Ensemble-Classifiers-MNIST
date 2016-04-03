from data import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
class DTC:
    def __init__(self,Data,depth,split='b',criterion='c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        if split=='b': s='best'
        else: s='random'
        self.clf  = tree.DecisionTreeClassifier(criterion=c,splitter=s,max_depth=depth)
        self.clf.fit(Data.train_set,Data.train_labels)
        self.score = self.clf.score(Data.test_set,Data.test_labels)
        print self.score

class RFC:
    def __init__(self,Data,NumOfEstimators, criterion = 'c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        self.clf  = RandomForestClassifier(n_estimators = NumOfEstimators,criterion = c, n_jobs = 4)
        self.clf.fit(Data.train_set,Data.train_labels)
        self.score = self.clf.score(Data.test_set,Data.test_labels)
        print self.score

class AB:
    def __init__(self,Data,NumOfEstimators):
        self.clf = AdaBoostClassifier(n_estimators = NumOfEstimators)
        self.clf.fit(Data.train_set,Data.train_labels)
        self.score1 = self.clf.score(Data.train_set[:1000],Data.train_labels[:1000])
        self.score2 = self.clf.score(Data.test_set, Data.test_labels)
        print NumOfEstimators,self.score1, self.score2

# D=Data()
# f=open('DataClassifiers.pkl','wb')
# import pickle
# pickle.dump(D,f,pickle.HIGHEST_PROTOCOL)
# f.close()

import pickle
f=open('DataClassifiers.pkl','rb')
D = pickle.load(f)
f.close()

for i in range(20,150,10):
    e=AB(D,i+1)
