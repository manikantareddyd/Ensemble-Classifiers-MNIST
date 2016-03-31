from data import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class DTC:
    def __init__(self,Data,depth,criterion='c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        self.clf  = tree.DecisionTreeClassifier(criterion=c,max_depth=depth)
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
