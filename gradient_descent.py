import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest


# create random data with help of uniform distrubition
x_learn=np.random.uniform(low=0.5, high=13.3, size=(1000,))
y_learn=x_learn*12

x_test=np.random.uniform(low=0.5, high=13.3, size=(200,))
y_test=x_test*12


# Graph output of this data is a linear func, so hypothesis func should look like;
# h(x)= Q(0) + Q(1)*x

# initialize values of q_0 and q_1
q_0=0
q_1=0

# empty list to append error values for each q_0 and q_1
cost=[]

# set proper learning rate
a=0.1

# create an empty figure 
plt.figure()

# calculate squared mean error
def cost_func(q_0,q_1):
    sum=0
    for (x,y) in zip_longest(x_learn,y_learn):
        sum=sum+(1/2000)*(q_0+q_1*x-y)**2
    cost.append(sum)

# calculate q_0,q_1    
def gradient_descent(a,x,y,q_0,q_1):
    q_0= q_0 - a*1/1000*(q_0+q_1*x-y)
    q_1= q_1 - a*1/1000*(q_0+q_1*x-y)*x
    return q_0,q_1

# test calculated values 
def test(x,y,q_0,q_1):
    pre_value= q_0 + q_1*x
    accuracy_score= (np.absolute(pre_value - y)/y)*100
    print(pre_value,y,accuracy_score)


# run learn algorithms
for (x,y) in zip_longest(x_learn,y_learn):
        q_0,q_1=gradient_descent(a,x,y,q_0,q_1)
        cost_func(q_0,q_1)

# run test algortihm
for (x,y) in zip_longest(x_test,y_test):
    test(x,y,q_0,q_1)

# plot the cost func
plt.plot(cost)
plt.xlabel('iterations')
plt.ylabel('cost values')
plt.title('Cost Function')
plt.show()







