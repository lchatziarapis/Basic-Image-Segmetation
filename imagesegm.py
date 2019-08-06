import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal

#Simple Project Image Segmetation using Expectation-Maximization Algorithm

K = int(input('Please enter K clustering number: '))

img = mpimg.imread('im.jpg')
formattedImg = np.array(img.reshape((img.shape[0]*img.shape[1],3)),dtype=np.int)

N,D = formattedImg.shape


def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def exp_trick(y):
	max_of_rows = np.max(y, 1)
	m = np.array([max_of_rows, ] * y.shape[1]).T
	y = y - m
	y = np.exp(y)
	return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T

def fitEM(x):

    # Initialize default parameters
    means = x[np.random.choice(N, K, False), :] # K x D
    covariances = [np.eye(D)] * K # K x K
    weights = [1./K] * K
    iter = 40
    tol = 1e-15
    parts = np.zeros((N,K))
    log_likelihoods = []
    print('Maximum iterations: ',iter)
    print('Epsilon Convergance: ',tol)
    for it in range(iter):

        for k in range(K):
            temp = multivariate_normal.pdf(x,means[k],covariances[k])
            temp[temp<=0] = 1e-100
            parts[:,k] = np.log(weights[k]) + np.log(temp)


        log_likelihood = log_sum_exp(parts)
        log_likelihoods.append(log_likelihood)

        ## Normalize so that the responsibility matrix is row stochastic
        parts = exp_trick(parts)

        ## The number of datapoints belonging to each gaussian
        N_ks = np.sum(parts, axis = 0)

        for k in range(K):
            means[k] = (1. / N_ks[k]) * np.sum(parts[:, k] * x.T, axis = 1).T
            x_mu = x - means[k]

            #covariances[k] = np.array( 1 / N_ks[k] * np.dot(np.multiply(x_mu.T, parts[:, k]), x_mu))
            covariances[k] =  np.sqrt( np.sum(parts[:,k]*(x-means[k]).T**2)/N_ks[k])

            weights[k] = N_ks[k]/N

        if len(log_likelihoods) < 2 : continue
        print('Iter: ',it,'error: ',np.abs(log_likelihood - log_likelihoods[-2]))
        if np.abs(log_likelihood - log_likelihoods[-2]) < tol: break
    return means,parts

newMeans,outR = fitEM(formattedImg)
newImage = np.zeros((N,D))
resp = np.argmax(outR,axis=1)

for n in range(N):
    newImage[n,:] = newMeans[resp[n]]
print('Reconstruction Error',np.sum(np.linalg.norm(formattedImg - newImage)**2)/N)

plt.imshow(np.array(newImage.reshape((img.shape[0],img.shape[1],3)),dtype=np.int))
plt.show()
