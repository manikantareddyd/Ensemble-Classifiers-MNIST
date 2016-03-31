from classifiers import *

Data = Data()
f=open('Results/DTC.txt','w')
for depth in range(1,10):
    r=DTC(Data,depth)
    f.write(str(depth)+' '+str(r.score)+'\n')

f.close()
