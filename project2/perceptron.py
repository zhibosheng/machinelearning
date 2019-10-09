import numpy as np
from matplotlib import *
#matplotlib.use('PDF')
from pylab import *
#</GRADED>
import sys
import matplotlib.pyplot as plt
import time

# add p02 folder
sys.path.insert(0, './p02/')


print('You\'re running python %s' % sys.version.split(' ')[0])
def perceptronUpdate(x,y,w):
    """
    function w=perceptronUpdate(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions
    
    Output:
    w : weight vector after updating (d)
    """
    assert(y in {-1,1})
    assert(len(w.shape)==1), "At the update w must be a vector not a matrix (try w=w.flatten())"
    assert(len(x.shape)==1), "At the update x must be a vector not a matrix (try x=x.flatten())"
    
    ## fill in code ...
    ## ... until here
    if y * np.matmul(w,x) <= 0:
        w = w + y*x
    return w.flatten()
x=rand(5) # random weight vector
w=rand(5) # random feature vector
y=-1 # random label
wnew=perceptronUpdate(x,y,w.copy()) # do a perceptron update
assert(norm(wnew-w+x)<1e-10), "perceptronUpdate didn't pass the test : (" # if correct, this should return 0
print("Looks like you passed the update test : )")

def perceptron(xs,ys):
    """
    function w=perceptron(xs,ys);
    
    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    b : bias term
    """

    assert(len(xs.shape)==2), "The first input to Perceptron must be a _matrix_ of row input vecdtors."
    assert(len(ys.shape)==1), "The second input to Perceptron must be a _vector_ of n labels (try ys.flatten())."
        
    n, d = xs.shape     # so we have n input vectors, of d dimensions each
    w = np.zeros((d,))
    b = 0
    ## fill in code ...
    ## ... until here
    for _ in range(100):
        for i in range(n):
            feature_vector = xs[i]
            label = ys[i]
            if label*(np.matmul(w,feature_vector)+b) <= 0:
                w += label*feature_vector
                b += label
    return (w,b)

def perceptron(xs,ys):
    """
    function w=perceptron(xs,ys);
    
    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    b : bias term
    """

    assert(len(xs.shape)==2), "The first input to Perceptron must be a _matrix_ of row input vecdtors."
    assert(len(ys.shape)==1), "The second input to Perceptron must be a _vector_ of n labels (try ys.flatten())."
        
    n, d = xs.shape     # so we have n input vectors, of d dimensions each
    w = np.zeros((d,))
    b = 0
    ## fill in code ...
    ## ... until here
    for _ in range(100):
        for i in range(n):
            feature_vector = xs[i]
            label = ys[i]
            if label*(np.matmul(w,feature_vector)+b) <= 0:
                w += label*feature_vector
                b += label
    return (w,b)

# number of input vectors
N = 100

# generate random (linarly separable) data
xs = np.random.rand(N, 2)*10-5

# defining random hyperplane
w0 = np.random.rand(2)
b0 = rand()*2-1;

# assigning labels +1, -1 labels depending on what side of the plane they lie on
ys = np.sign(xs.dot(w0)+b0)

# call perceptron to find w from data
w,b = perceptron(xs.copy(),ys.copy())

# test if all points are classified correctly
assert (all(np.sign(ys*(xs.dot(w)+b))==1.0))  # yw'x should be +1.0 for every input
print("Looks like you passed the Perceptron test! :o)")

# we can make a pretty visualizxation
from helperfunctions import visboundary
visboundary(w,b,xs,ys)

def onclick(event):
    global w,b,ldata,ax,line,xydata

    pos=np.array([[event.xdata],[event.ydata]])
    if event.key == 'shift': # add positive point
        color='or'
        label=1
    else: # add negative point
        color='ob'
        label=-1    
    ax.plot(pos[0],pos[1],color)
    ldata.append(label);
    xydata=np.vstack((xydata,pos.T))
    
    # call Perceptron function
    w,b=perceptron(xydata,np.array(ldata).flatten())

    # draw decision boundary
    q=-b/(w**2).sum() *w;
    if line==None:
        line, = ax.plot([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]],'b--')
    else:
        line.set_data([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]])
        


xydata=rand(0,2)
ldata=[]
w=zeros(2)
b=0
line=None

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(0,1)
plt.ylim(0,1)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
title('Use shift-click to add negative points.')
w,b=perceptron(xydata,np.array(ldata).flatten())
q=-b/(w**2).sum() *w;
line, = ax.plot([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]],'b--')
line,=ax.plot([0.2,0.2],[0.8,0.8])
def classifyLinear(xs,w,b):
    """
    function preds=classifyLinear(xs,w,b)
    
    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)
    
    Output:
    preds: predictions (1xn)
    """    
    w = w.flatten()    
    predictions=np.zeros(xs.shape[0])
    ## fill in code ...
    for i in range(xs.shape[0]):
        predictions[i] = np.matmul(w,xs[i]) + b
    ## ... until here
    return predictions

xs=rand(1000,2)-0.5 # draw random data 
w0=np.array([0.5,-0.3]) # define a random hyperplane 
b0=-0.1 # with bias -0.1
ys=np.sign(xs.dot(w0)+b0) # assign labels according to this hyperplane (so you know it is linearly separable)
assert (all(np.sign(ys*classifyLinear(xs,w0,b0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly
print("Looks like you passed the classifyLinear test! :o)")