import mpi4py
from mpi4py import MPI 
import time

digit_R=[[0,0,0,0,0,0,0],
         [1,1,0,0,0,0,0],
         [1,0,1,1,0,1,1],
         [1,1,1,0,0,1,1],
         [0,1,0,0,1,0,1],
         [0,1,1,0,1,1,1],
         [0,1,1,1,1,1,1],
         [1,1,0,0,0,0,0],
         [1,1,1,1,1,1,1],
         [1,1,1,0,1,1,1]]

y = [[1,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0],
     [0,0,0,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,1]]

# Make a prediction with weights
def predict(row,weights):
	activation = weights[0]
	#print(len(row))
	for i in range(len(row)):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train,y, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0])+1)]
	cost=[]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row,i in zip(train,y):
			prediction = predict(row, weights)
			error = i - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error   #bias update
			for i in range(len(row)):
				weights[i + 1] = weights[i+1] + l_rate * error * row[i]   #weight update 
		#cost.extend([sum_error])
	return weights


def perceptron_learning_algorithm(logic,y,name) :
    l_rate = 0.1
    n_epoch = 30
    print("-----------------------%s---------------------" %(name))
    weights = train_weights(logic,y,l_rate, n_epoch)
    print("Trained weight = %s" %(weights))
    for row,i in zip(logic,y):
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (i, prediction))
    return(weights)

start_time = time.time()
for i in range(10):
    perceptron_learning_algorithm(digit_R,y[i],i)
stop_time = time.time()
print("Time consumed = ",stop_time - start_time)