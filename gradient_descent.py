import numpy as np
import matplotlib.pyplot as plt
from itertools import count, zip_longest


# create random data with help of uniform distrubition
x_learn=np.random.uniform(low=0.5, high=13.3, size=(1000,))
y_learn=x_learn*12

x_test=np.random.uniform(low=0.5, high=13.3, size=(200,))
y_test=x_test*12


''' 
Graph output of this data is a linear func, so hypothesis func should look like;
# h(x)= Q(0) + Q(1)*x
'''

# initialize parameters q_0 and q_1
q_0,q_1=0,0 

# empty list to append error values for each q_0 and q_1
cost=[]

# set proper learning rate
a=0.01

# create an empty figure 
plt.figure()

# calculate squared mean error
def cost_func(q_0,q_1):
    sum=0
    for (x,y) in zip_longest(x_learn,y_learn):
        sum=sum+(1/2000)*(q_0+q_1*x-y)**2
    cost.append(sum)

# calculate q_0,q_1    
def grad(q_0,q_1):
    derivative_0=0
    derivative_1=0
    for (x,y) in zip_longest(x_learn,y_learn):
        derivative_0 = derivative_0 + 1/1000*(q_0+q_1*x-y)
        derivative_1 = derivative_1+ 1/1000*(q_0+q_1*x-y)*x
    list=[derivative_0,derivative_1]
    return list

# test calculated values 
def test(x,y,q_0,q_1):
    pre_value= q_0 + q_1*x
    accuracy_score= int(100-(np.absolute(pre_value - y)/y)*100)
    print(pre_value,y,accuracy_score)
    return accuracy_score

# run learning algorithm
def descent(q_0,q_1,a):
    count=0
    while True:
        count += 1
        print(count)
        q_0 = q_0-a*grad(q_0,q_1)[0]
        q_1 = q_1-a*grad(q_0,q_1)[1]
        print(q_0,q_1)
        cost_func(q_0,q_1)
        if count>1000:
            break
    return q_0,q_1

q_0,q_1=descent(q_0,q_1,a)

# run test algortihm
acc=0
for (x,y) in zip_longest(x_test,y_test):
    acc=acc+test(x,y,q_0,q_1)
acc=acc/200

# plot the cost func
plt.plot(cost)
plt.xlabel('iterations')
plt.ylabel('cost values')
plt.title('Cost Function')
plt.show()
print(f'q_0:{q_0},q_1:{q_1},mean accuracy:{acc}')







