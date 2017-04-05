import numpy as np

def nonlin(x,deriv=0):
    if(deriv==1):
        return x*(1-x)
  
    return 1/(1+np.exp(-x))


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
inp=len(X[0])
h1=100
h2=70
output=1

np.random.seed(1)
# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((inp,h1)) - 1
syn1 = 2*np.random.random((h1,h2)) - 1
syn2 = 2*np.random.random((h2,output)) - 1

for j in range(100000):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    
    # how much did we miss the target value?
    l3_error = y - l3
    
    if (j% 1000) == 0:
        print ("Error:" + str(np.mean(np.abs(l3_error))))
        
    #back-propagation through layers
    l3_delta = l3_error*nonlin(l3,deriv=1)
    l2_error = l3_delta.dot(syn2.T)    
    l2_delta = l2_error*nonlin(l2,deriv=1)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=1)

    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

#test neural network
testX=[1, 0, 0]
res= nonlin(np.dot(nonlin(np.dot( nonlin(np.dot(testX,syn0)),syn1)),syn2))
print (res)






