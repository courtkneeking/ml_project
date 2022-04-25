import numpy as np
import math
import matplotlib.pyplot as plt

def plotDistributions(Data):
    for x in Data: 
        plt.hist(x)
        plt.show()
        # try using various bin sizes, intervals (list) gets the proper range and x-values for plot;
        # starts from the min value in the distribution rounded down to the nearest BIN_SIZE;
        # ends at the max value rounded up to the nearest BIN_SIZE;
        # BIN_SIZE = 25
        # intervals = [i*BIN_SIZE for i in range(int(math.floor(min(city)/BIN_SIZE)), int(math.ceil(max(city)/BIN_SIZE))+1)]
        # plt.hist(city, bins=intervals)
        

# Question 2. function learnParams 
# input: a data set 
# returns: the learned mean, standard deviation for each class. 
def learnParams(Data):
    # separate by class labels, c0 for label=0, etc. 
    c0 = [x[1] for x in Data if x[0] == 0]
    c1 = [x[1] for x in Data if x[0] == 1]
    # get mean for each class label 
    mean_c0=sum(c0)/len(c0)
    mean_c1=sum(c1)/len(c1)
    # get variance, subtract mean from each value & square(result)
    # sum variance array, divide by |data|; 
    v_c0=sum([(x-mean_c0)**2 for x in c0])/(len(c0))
    v_c1=sum([(x-mean_c1)**2 for x in c1])/(len(c1))
    # standard deviation is just the sqrt of variance 
    # it seems like the answer should be rounded to 2nd significant digit
    std_c0 = round(math.sqrt(v_c0),2)
    std_c1 = round(math.sqrt(v_c1),2)
    return np.array([[mean_c0, std_c0], [mean_c1, std_c1]])

# Question 3. 
# function learnPriors that takes in a data set and 
# returns the prior probability of each class. 
def learnPriors(Data):
    # separate by class labels, c0 for label=0, etc. 
    c0 = [x[1] for x in Data if x[0] == 0]
    c1 = [x[1] for x in Data if x[0] == 1]
    # how many of each label relative to |Data| -> the priors 
    p_c0 = len(c0)/len(Data)
    p_c1 = len(c1)/len(Data)
    return np.array([p_c0,p_c1])
    
# Helper for labelBayes 
def Gauss(x, mean, std):
    exponent = math.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

# Question 4
# return max  P(yi|X=x) for yi in Y (Y is all classes)
# P(yi | X=x) = P(X=x | yi)* P(yi)
# use paramsL to compute likelihoods P(X=x|yi) 
def labelBayes(postTimes, paramsL, priors):
    labelsOut = []
    # for xi in X (X is the posting times)
    for x in postTimes: 
        likelihoods = []
        # for yj in Y (Y is all classes), X=xi
        for ((mean, std), prior) in zip(paramsL, priors): 
            # use paramsL to compute likelihoods P(X=x|yj) 
            likelihood = Gauss(x, mean,std) 
            prior*likelihood
            likelihoods.append(likelihood) # append for each class 
        # predict: max class label is index of max value
        maxLikelihood = likelihoods.index(max(likelihoods)) 
        # add to labelsOut, it will automatically align with that of x 
        labelsOut.append(maxLikelihood)  
    return np.array(labelsOut)