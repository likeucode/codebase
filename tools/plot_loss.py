# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:56:46 2017

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt  
def parse_log(file_path,flag):
    file =open(file_path)  
    filelines=file.readlines()  
    print len(filelines)  
    Iters=[]  
    Loss=[]  
    temp=[]  
    for i  in range (1,len(filelines)):  
        line=filelines[i].split(' ')  
        #print line  
        for j in range(0,len(line)):  
            if line[j] !='':  
                #print line[j]  
               temp.append(line[j])  
  
    for i in range(0,len(temp)):  
        if i%4==0:  
            Iters.append(int(temp[i]))  
        
        if flag:
            if i%4==3:  
                Loss.append(float(temp[i])) 
        else:
            if i%4==2:
                Loss.append(float(temp[i]))
    file.close()
    return Iters,Loss
    
    
    
train_file='C:/Users/User/Desktop/tmp/resnet.log.train'
test_file='C:/Users/User/Desktop/tmp/resnet.log.test'
iters_train,train_loss=parse_log(train_file,0)
iters_test,test_loss=parse_log(test_file,1)
#print Iters  
#print TrainingLoss  
  
plt.plot(iters_train,train_loss, 'b')
plt.plot(iters_test,test_loss, 'r')
  
plt.title('Train and test loss  VS Iters')  

plt.xlabel('Iters')  
plt.ylabel('loss')  
plt.xticks(range(0,max(iters_test),50000))
plt.xticks(rotation=48)
plt.yticks(np.linspace(0,10,12)) 
plt.legend(loc="lower right")
plt.savefig('loss.png',dpi=200)  
plt.show()  

#plt.title('Testloss  VS Iters') 
#plt.ylabel('Testloss')  
#plt.savefig('testloss.png',dpi=200)   
