from classifiers import *

Data = Data()
f=open('Results/DTC.txt','w')
for depth in range(10,11):
    r=RFC(Data,depth)
    f.write(str(depth)+' '+str(r.score)+'\n')

f.close()
