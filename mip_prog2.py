import mpi4py
from mpi4py import MPI 
import numpy as np
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
def predict(row,weights,rank,comm):
      
    a0 = weights[0]
    
    w = np.array(weights[1:len(row)+1])
    r = np.array(row[0:len(row)])
    a = w*r    

    a1 =  comm.allreduce(a,op=MPI.SUM)
    
    
    a = sum(a1) + a0

    return 1.0 if a >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train,y, l_rate, n_epoch,rank,comm):
	weights = [0.0 for i in range(len(train[0])+1)]
	cost=[]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row,i in zip(train,y):
			prediction = predict(row, weights,rank,comm)
			error = i - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error   #bias update
			for i in range(len(row)):
				weights[i + 1] = weights[i+1] + l_rate * error * row[i]   #weight update 
		#cost.extend([sum_error])
	return weights


def perceptron_learning_algorithm(logic,y,name,rank,comm) :
    l_rate = 0.1
    n_epoch = 30
    print("-----------------------%s---------------------" %(name))
    weights = train_weights(logic,y,l_rate, n_epoch,rank,comm)
    print("Trained weight = %s" %(weights))
    for row,i in zip(logic,y):
        prediction = predict(row, weights,rank,comm)
        print("Expected=%d, Predicted=%d" % (i, prediction))
    return(weights)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    size = comm.Get_size()
    print(rank,size)
    
    start_time = time.time()
    for i in range(10):
        perceptron_learning_algorithm(digit_R,y[i],i,rank,comm)
    stop_time = time.time()
    print("Time consumed = ",stop_time - start_time)