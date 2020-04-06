from collections import namedtuple
import math
import numba
import numpy as np
import pandas as pd
import random
import sys



@numba.jit(nopython=True)
def fetch_pair(emx, i, j, min_expression, max_expression):
    # extract pairwise data
    x = emx[i]
    y = emx[j]

    # initialize labels
    labels = np.zeros(len(x), dtype=np.int8)

    # mark thresholded samples
    labels[(x < min_expression) | (y < min_expression)] = -6
    labels[(x > max_expression) | (y > max_expression)] = -6

    # mark nan samples
    labels[np.isnan(x) | np.isnan(y)] = -9

    return (x, y, labels)



@numba.jit(nopython=True)
def mark_outliers(x, y, labels, k, marker):
    # extract samples in cluster k
    mask = (labels == k)
    x_sorted = np.copy(x[mask])
    y_sorted = np.copy(y[mask])

    # make sure cluster is not empty
    if len(x_sorted) == 0 or len(y_sorted) == 0:
        return

    # sort arrays
    x_sorted.sort()
    y_sorted.sort()

    # compute quartiles and thresholds for each axis
    n = len(x_sorted)

    Q1_x = x_sorted[n * 1 // 4]
    Q3_x = x_sorted[n * 3 // 4]
    T_x_min = Q1_x - 1.5 * (Q3_x - Q1_x)
    T_x_max = Q3_x + 1.5 * (Q3_x - Q1_x)

    Q1_y = y_sorted[n * 1 // 4]
    Q3_y = y_sorted[n * 3 // 4]
    T_y_min = Q1_y - 1.5 * (Q3_y - Q1_y)
    T_y_max = Q3_y + 1.5 * (Q3_y - Q1_y)

    # mark outliers
    for i in range(len(labels)):
        if labels[i] == k:
            outlier_x = (x[i] < T_x_min or T_x_max < x[i])
            outlier_y = (y[i] < T_y_min or T_y_max < y[i])

            if outlier_x or outlier_y:
                labels[i] = marker



@numba.jit(nopython=True)
def vector_diff_norm(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



GMM = namedtuple('GMM', [
    'data',
    'labels',
    'pi',
    'mu',
    'sigma',
    'sigmaInv',
    'normalizer',
    'MP',
    'counts',
    'logpi',
    'gamma',
    'logL',
    'entropy'
])



@numba.jit(nopython=True)
def gmm_initialize_components(gmm, X, N, K):
    # initialize random state
    random.seed(1)

    # initialize each mixture component
    for k in range(K):
        # initialize mixture weight to uniform distribution
        gmm.pi[k] = 1.0 / K
        
        # initialize mean to a random sample from X
        i = random.randrange(N)
        
        gmm.mu[k] = X[i]
        
        # initialize covariance to identity matrix
        gmm.sigma[k] = np.identity(2)



@numba.jit(nopython=True)
def gmm_prepare_components(gmm, K):
    D = 2

    for k in range(K):
        # compute precision matrix (inverse of covariance matrix)
        # det = matrix_inverse(gmm.sigma[k], gmm.sigmaInv[k])
        A = gmm.sigma[k]
        B = gmm.sigmaInv[k]

        # compute determinant of covariance matrix
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

        # return failure if matrix inverse failed
        if det <= 0.0 or np.isnan(det):
            return False

        # compute precision matrix
        B[0, 0] = +A[1, 1] / det
        B[0, 1] = -A[0, 1] / det
        B[1, 0] = -A[1, 0] / det
        B[1, 1] = +A[0, 0] / det

        # compute normalizer term for multivariate normal distribution
        gmm.normalizer[k] = -0.5 * (D * math.log(2.0 * math.pi) + math.log(det))

    return True



@numba.jit(nopython=True)
def gmm_initialize_means(gmm, X, N, K):
    max_iterations = 20
    tolerance = 1e-3
    
    # initialize workspace
    MP = gmm.MP
    counts = gmm.counts
    
    for t in range(max_iterations):
        # compute mean and sample count for each component
        MP[:] = 0
        counts[:] = 0
    
        for i in range(N):
            # determine the component mean which is nearest to x_i
            min_dist = math.inf
            min_k = 0    
            for k in range(K):
                dist = vector_diff_norm(X[i], gmm.mu[k])
                if min_dist > dist:
                    min_dist = dist
                    min_k = k

            # update mean and sample count
            MP[min_k] += X[i]
            counts[min_k] += 1

        # scale each mean by its sample count
        for k in range(K):
            MP[k] /= counts[k]

        # compute the total change of all means
        diff = 0.0

        for k in range(K):
            diff += vector_diff_norm(MP[k], gmm.mu[k])

        diff /= K
    
        # update component means
        for k in range(K):
            gmm.mu[k] = MP[k]
        
        # stop if converged
        if diff < tolerance:
            break



@numba.jit(nopython=True)
def gmm_compute_estep(gmm, X, N, K):
    # compute logpi
    for k in range(K):
        gmm.logpi[k] = math.log(gmm.pi[k])

    # compute the log-probability for each component and each point in X
    logProb = gmm.gamma
    
    for i in range(N):
        for k in range(K):
            # compute xm = (x - mu)
            xm = X[i] - gmm.mu[k]
            
            # compute Sxm = Sigma^-1 xm
            Sxm = np.dot(gmm.sigmaInv[k], xm)

            # compute xmSxm = xm^T Sigma^-1 xm
            xmSxm = np.dot(xm, Sxm)
            
            # compute log(P) = normalizer - 0.5 * xm^T * Sigma^-1 * xm
            logProb[i, k] = gmm.normalizer[k] - 0.5 * xmSxm

    # compute gamma and log-likelihood
    logL = 0.0
    
    for i in range(N):
        # compute a = argmax(logpi_k + logProb_ik, k)
        maxArg = -math.inf
        for k in range(K):
            arg = gmm.logpi[k] + logProb[i, k]
            if maxArg < arg:
                maxArg = arg

        # compute logpx
        sum_ = 0.0
        for k in range(K):
            sum_ += math.exp(gmm.logpi[k] + logProb[i, k] - maxArg)

        logpx = maxArg + math.log(sum_)

        # compute gamma_ik
        for k in range(K):
            gmm.gamma[i, k] += gmm.logpi[k] - logpx
            gmm.gamma[i, k] = math.exp(gmm.gamma[i, k])

        # update log-likelihood
        logL += logpx

    # return log-likelihood
    return logL



@numba.jit(nopython=True)
def gmm_compute_mstep(gmm, X, N, K):
    for k in range(K):
        # compute n_k = sum(gamma_ik)
        n_k = 0.0
        
        for i in range(N):
            n_k += gmm.gamma[i, k]

        # update mixture weight
        gmm.pi[k] = n_k / N

        # update mean
        mu = np.zeros((2,))

        for i in range(N):
            mu += gmm.gamma[i, k] * X[i]

        mu /= n_k
        
        gmm.mu[k] = mu

        # update covariance matrix
        sigma = np.zeros((2, 2))
        
        for i in range(N):
            # compute xm = (x_i - mu_k)
            xm = X[i] - mu

            # compute Sigma_ki = gamma_ik * (x_i - mu_k) (x_i - mu_k)^T
            sigma += gmm.gamma[i, k] * np.dot(xm.T, xm)

        sigma /= n_k

        gmm.sigma[k] = sigma



@numba.jit(nopython=True)
def gmm_compute_labels(gamma, N, K, labels):
    for i in range(N):
        # determine the value k for which gamma_ik is highest
        max_k = -1
        max_gamma = -math.inf

        for k in range(K):
            if max_gamma < gamma[i, k]:
                max_k = k
                max_gamma = gamma[i, k]

        # assign x_i to cluster k
        labels[i] = max_k



@numba.jit(nopython=True)
def gmm_compute_entropy(gamma, N, labels):
    E = 0.0
    
    for i in range(N):
        k = labels[i]
        E -= math.log(gamma[i, k])
    
    return E



@numba.jit(nopython=True)
def gmm_fit(gmm, X, N, K, labels):
    # initialize mixture components
    gmm_initialize_components(gmm, X, N, K)

    # initialize means with k-means
    gmm_initialize_means(gmm, X, N, K)

    # run EM algorithm
    max_iterations = 100
    tolerance = 1e-8
    prevLogL = -math.inf
    currLogL = -math.inf

    for t in range(max_iterations):
        # pre-compute precision matrix and normalizer term for each mixture component
        success = gmm_prepare_components(gmm, K)

        # return failure if matrix inverse failed
        if not success:
            return False

        # perform E step
        prevLogL = currLogL
        currLogL = gmm_compute_estep(gmm, X, N, K)

        # check for convergence
        if abs(currLogL - prevLogL) < tolerance:
            break

        # perform M step
        gmm_compute_mstep(gmm, X, N, K)

    # save outputs
    gmm.logL[0] = currLogL
    gmm_compute_labels(gmm.gamma, N, K, labels)
    gmm.entropy[0] = gmm_compute_entropy(gmm.gamma, N, labels)

    return True



@numba.jit(nopython=True)
def compute_aic(K, D, logL):
    p = K * (1 + D + D * D)
    
    return 2 * p - 2 * logL



@numba.jit(nopython=True)
def compute_bic(K, D, logL, N):
    p = K * (1 + D + D * D)
    
    return math.log(N) * p - 2 * logL



@numba.jit(nopython=True)
def compute_icl(K, D, logL, N, E):
    p = K * (1 + D + D * D)

    return math.log(N) * p - 2 * logL + 2 * E



@numba.jit(nopython=True)
def gmm_compute(x, y, labels, min_samples, min_clusters, max_clusters, criterion):
    # extract pairwise data
    mask = (labels == 0)
    gmm_data = np.vstack((x, y)).T[mask]
    gmm_labels = np.copy(labels[mask])

    # initialize gmm
    N = len(gmm_labels)
    K = max_clusters

    gmm = GMM(
        data       = gmm_data,
        labels     = gmm_labels,
        pi         = np.empty((K,)),
        mu         = np.empty((K, 2)),
        sigma      = np.empty((K, 2, 2)),
        sigmaInv   = np.empty((K, 2, 2)),
        normalizer = np.empty((K,)),
        MP         = np.empty((K, 2)),
        counts     = np.empty((K,)),
        logpi      = np.empty((K,)),
        gamma      = np.empty((N, K)),
        logL       = np.array([0.0]),
        entropy    = np.array([0.0])
    )

    # perform clustering only if there are enough samples
    bestK = 0

    if N >= min_samples:
        # determine the number of clusters
        bestValue = math.inf

        for K in range(min_clusters, max_clusters + 1):
            # run the clustering model
            success = gmm_fit(gmm, gmm.data, N, K, gmm.labels)

            if not success:
                continue

            # compute the criterion value of the model
            value = math.inf

            if criterion == 'aic':
                value = compute_aic(K, 2, gmm.logL[0])
            elif criterion == 'bic':
                value = compute_bic(K, 2, gmm.logL[0], N)
            elif criterion == 'icl':
                value = compute_icl(K, 2, gmm.logL[0], N, gmm.entropy[0])

            # save the model with the lowest criterion value
            if value < bestValue:
                bestK = K
                bestValue = value
                labels[mask] = gmm.labels

    return bestK, labels



@numba.jit(nopython=True)
def pearson(x, y):
    n = len(x)
    sumx = 0.0
    sumy = 0.0
    sumx2 = 0.0
    sumy2 = 0.0
    sumxy = 0.0
    
    for i in range(n):
        x_i = x[i]
        y_i = y[i]
        
        sumx += x_i
        sumy += y_i
        sumx2 += x_i * x_i
        sumy2 += y_i * y_i
        sumxy += x_i * y_i

    return (n*sumxy - sumx*sumy) / math.sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy))



@numba.jit(nopython=True)
def compute_rank(array):
    n = len(array)
    i = 0

    while i < n - 1:
        a_i = array[i]

        if a_i == array[i + 1]:
            j = i + 2
            rank = 0.0

            # we have detected a tie, find number of equal elements
            while j < n and a_i == array[j]:
                j += 1

            # compute rank
            for k in range(i, j):
                rank += k

            # divide by number of ties
            rank /= (j - i)

            for k in range(i, j):
                array[k] = rank

            i = j
        else:
            # no tie - set rank to natural ordered position
            array[i] = i
            i += 1

    if i == n - 1:
        array[n - 1] = (n - 1)



@numba.jit(nopython=True)
def spearman(x, y):
    n = len(x)
    x_rank = np.copy(x)
    y_rank = np.copy(y)
    
    x_argsort = np.argsort(x_rank)
    x_rank = x_rank[x_argsort]
    y_rank = y_rank[x_argsort]
    compute_rank(x_rank)

    y_argsort = np.argsort(y_rank)
    x_rank = x_rank[y_argsort]
    y_rank = y_rank[y_argsort]
    compute_rank(y_rank)

    return pearson(x_rank, y_rank)



@numba.jit(nopython=True)
def compute_correlation(x, y, labels, k, method, min_samples):
    # extract samples in cluster k
    x_k = x[labels == k]
    y_k = y[labels == k]

    # make sure there are enough samples
    if len(x_k) < min_samples:
        return np.nan

    # compute correlation
    if method == 'pearson':
        return pearson(x_k, y_k)
    elif method == 'spearman':
        return spearman(x_k, y_k)
    else:
        return np.nan



@numba.jit(nopython=True)
def similarity_cpu(
    emx,
    clusmethod,
    corrmethod,
    preout,
    postout,
    minexpr,
    maxexpr,
    minsamp,
    minclus,
    maxclus,
    criterion,
    mincorr,
    maxcorr):

    cmx = []

    for i in range(10): # range(emx.shape[0]):
        for j in range(i):
            # fetch pairwise input data
            x, y, labels = fetch_pair(emx, i, j, minexpr, maxexpr)

            # remove pre-clustering outliers
            if preout:
                mark_outliers(x, y, labels, 0, -7)

            # perform clustering
            K = 1

            if clusmethod == 'gmm':
                K, labels = gmm_compute(x, y, labels, minsamp, minclus, maxclus, criterion)

            # remove post-clustering outliers
            if K > 1 and postout:
                for k in range(K):
                    mark_outliers(x, y, labels, k, -8)

            # perform correlation
            correlations = np.array([compute_correlation(x, y, labels, k, corrmethod, minsamp) for k in range(K)])

            # save correlation matrix
            valid = np.array([(~np.isnan(corr) and mincorr <= abs(corr) and abs(corr) <= maxcorr) for corr in correlations])
            num_clusters = valid.sum()
            cluster_idx = 1

            for k in range(K):
                corr = correlations[k]

                # make sure correlation, p-value meets thresholds
                if valid[k]:
                    # compute sample mask
                    y_k = np.copy(labels)
                    y_k[(y_k >= 0) & (y_k != k)] = 0
                    y_k[y_k == k] = 1
                    y_k[y_k < 0] *= -1

                    # compute summary statistics
                    num_samples = (y_k == 1).sum()

                    # append correlation to cmx
                    cmx.append((i, j, cluster_idx, num_clusters, num_samples, corr, y_k))

                    # increment cluster index
                    cluster_idx += 1

    return cmx


                    
def main(use_numba=False):
    # define input parameters
    args_input = 'Yeast-1000.emx.txt'
    args_output = 'Yeast-1000.cmx.txt'
    args_clusmethod = 'gmm'
    args_corrmethod = 'spearman'
    args_preout = True
    args_postout = True
    args_minexpr = 0.0
    args_maxexpr = 20.0
    args_minsamp = 30
    args_minclus = 1
    args_maxclus = 5
    args_criterion = 'icl'
    args_mincorr = 0.5
    args_maxcorr = 1.0

    # load input data
    emx = pd.read_csv(args_input, sep='\t', index_col=0)

    # extract raw expression data
    emx_data = emx.values

    # compute similarity of each pair
    similarity = similarity_cpu if use_numba else similarity_cpu.py_func

    cmx = similarity(
        emx_data,
        args_clusmethod,
        args_corrmethod,
        args_preout,
        args_postout,
        args_minexpr,
        args_maxexpr,
        args_minsamp,
        args_minclus,
        args_maxclus,
        args_criterion,
        args_mincorr,
        args_maxcorr)
    
    # save cmx to output file
    output = open(args_output, 'w')

    for (i, j, cluster_idx, num_clusters, num_samples, corr, y_k) in cmx:
        sample_mask = ''.join([str(y_i) for y_i in y_k])

        output.write('%d\t%d\t%d\t%d\t%d\t%0.8f\t%s\n' % (i, j, cluster_idx, num_clusters, num_samples, corr, sample_mask))



if __name__ == '__main__':
    main(use_numba=int(sys.argv[1]))
