from data import *
from math import *
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn import tree
class DTC:
    def __init__(self,train_set,train_labels,depth,criterion='c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        self.clf  = tree.DecisionTreeClassifier(criterion=c,max_depth=depth)
        self.clf.fit(train_set,train_labels)

class AdaBoost:
    def __init__(self, Data, NoOfClassifiers = 10):
        self.alpha = []
        self.models = []
        self.predictions=[]
        self.Boost(Data, NoOfClassifiers)
    def loss(self,j,i,true_Label):
        M=[sum([self.alpha[t]*self.predictions[t][i] for t in range(tt)]) for tt in range(j)]
        return sum([exp(-1.0*true_Label*M[tt]) for tt in range(j)])
    def Boost(self,Data, NoOfClassifiers):
        L = len(Data.learning_set)
        p = [1/1.0*L for i in range(L)]

        for j in range(0,NoOfClassifiers):
            X_train, X_test, y_train, y_test = train_test_split(Data.learning_set, Data.learning_set_labels, test_size=0.1, random_state=datetime.now().second)
            clf = DTC(X_train,y_train, depth = j+1)
            predictions = clf.clf.predict(Data.learning_set)
            self.predictions.append(predictions)
            self.models.append(clf)
            error = 0
            print "Hm"
            for i in range(L):
                error += p[i]*self.loss(j,i,Data.learning_set_labels[i])*1.0
            print "Hm2" , error
            if error > 0.5:
                NoOfClassifiers = j-1
                break
            else:
                self.alpha.append(0.5*log(1-error/1.0*error))
                for i in range(L):
                    if predictions[i] == Data.learning_set_labels[i]:
                        p[i] = p[i]*sqrt(1/1.0*self.alpha[j])
                    else:
                        p[i] = p[i]*sqrt(1.0*self.alpha[j])

                    z = sum(p)
                    p = [i/1.0*z for i in p]
