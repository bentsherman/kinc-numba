from collections import namedtuple
import math
import numba
from numba import cuda
import numpy as np
import pandas as pd
import random
import sys



CLUSMETHOD_NONE = 1
CLUSMETHOD_GMM = 2

CRITERION_AIC = 1
CRITERION_BIC = 2
CRITERION_ICL = 3

CORRMETHOD_PEARSON = 1
CORRMETHOD_SPEARMAN = 2



@numba.jit(nopython=True)
def fetch_pair(x, y, min_expression, max_expression, labels):
    # label the pairwise samples
    N = 0

    for i in range(len(x)):
        # label samples with missing values
        if math.isnan(x[i]) or math.isnan(y[i]):
            labels[i] = -9

        # label samples which are below the minimum expression threshold
        elif x[i] < min_expression or y[i] < min_expression:
            labels[i] = -6

        # label samples which are above the maximum expression threshold
        elif x[i] > max_expression or y[i] > max_expression:
            labels[i] = -6

        # label any remaining samples as cluster 0
        else:
            N += 1
            labels[i] = 0

    # return number of clean samples
    return N




@numba.jit(nopython=True)
def next_power_2(n):
    pow2 = 2
    while pow2 < n:
        pow2 *= 2
    return pow2



@numba.jit(nopython=True)
def swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp



@numba.jit(nopython=True)
def bitonic_sort(array):
    n = len(array)
    bsize = n // 2

    ob = 2
    while ob <= n:
        ib = ob
        while ib >= 2:
            t = ib // 2
            for i in range(bsize):
                dir_ = -((i//(ob//2)) & 0x1)
                a = (i//t) * ib + (i%t)
                b = a + t
                if (not dir_ and (array[a] > array[b])) or (dir_ and (array[a] < array[b])):
                    swap(array, a, b)
            ib //= 2
        ob *= 2



@numba.jit(nopython=True)
def bitonic_sort_ff(array, extra):
    n = len(array)
    bsize = n // 2

    ob = 2
    while ob <= n:
        ib = ob
        while ib >= 2:
            t = ib // 2
            for i in range(bsize):
                dir_ = -((i//(ob//2)) & 0x1)
                a = (i//t) * ib + (i%t)
                b = a + t
                if (not dir_ and (array[a] > array[b])) or (dir_ and (array[a] < array[b])):
                    swap(array, a, b)
                    swap(extra, a, b)
            ib //= 2
        ob *= 2



@numba.jit(nopython=True)
def mark_outliers(x, y, labels, k, marker, x_sorted, y_sorted):
    # extract samples in cluster k
    n = 0
    
    for i in range(len(x)):
        if labels[i] == k:
            x_sorted[n] = x[i]
            y_sorted[n] = y[i]
            n += 1

    for i in range(n, len(x_sorted)):
        x_sorted[i] = math.inf
        y_sorted[i] = math.inf

    # make sure cluster is not empty
    if n == 0:
        return 0

    # sort arrays
    bitonic_sort(x_sorted)
    bitonic_sort(y_sorted)

    # compute quartiles and thresholds for each axis
    Q1_x = x_sorted[n * 1 // 4]
    Q3_x = x_sorted[n * 3 // 4]
    T_x_min = Q1_x - 1.5 * (Q3_x - Q1_x)
    T_x_max = Q3_x + 1.5 * (Q3_x - Q1_x)

    Q1_y = y_sorted[n * 1 // 4]
    Q3_y = y_sorted[n * 3 // 4]
    T_y_min = Q1_y - 1.5 * (Q3_y - Q1_y)
    T_y_max = Q3_y + 1.5 * (Q3_y - Q1_y)

    # mark outliers
    n = 0

    for i in range(len(labels)):
        if labels[i] == k:
            outlier_x = (x[i] < T_x_min or T_x_max < x[i])
            outlier_y = (y[i] < T_y_min or T_y_max < y[i])

            if outlier_x or outlier_y:
                labels[i] = marker
            elif labels[i] >= 0:
                n += 1

    # return number of remaining samples
    return n



@numba.jit(nopython=True)
def vector_assign(a, b):
    a[0] = b[0]
    a[1] = b[1]



@numba.jit(nopython=True)
def vector_add(a, b):
    a[0] += b[0]
    a[1] += b[1]



@numba.jit(nopython=True)
def vector_add_scaled(a, c, b):
    a[0] += c * b[0]
    a[1] += c * b[1]



@numba.jit(nopython=True)
def vector_subtract(a, b):
    a[0] -= b[0]
    a[1] -= b[1]



@numba.jit(nopython=True)
def vector_scale(a, c):
    a[0] *= c
    a[1] *= c



@numba.jit(nopython=True)
def vector_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]



@numba.jit(nopython=True)
def vector_diff_norm(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



@numba.jit(nopython=True)
def matrix_init_identity(M):
    M[0, 0] = 1
    M[0, 1] = 0
    M[1, 0] = 0
    M[1, 1] = 1



@numba.jit(nopython=True)
def matrix_scale(A, c):
    A[0, 0] *= c
    A[0, 1] *= c
    A[1, 0] *= c
    A[1, 1] *= c



@numba.jit(nopython=True)
def matrix_determinant(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]



@numba.jit(nopython=True)
def matrix_inverse(A, B, det):
    B[0, 0] = +A[1, 1] / det
    B[0, 1] = -A[0, 1] / det
    B[1, 0] = -A[1, 0] / det
    B[1, 1] = +A[0, 0] / det



@numba.jit(nopython=True)
def matrix_product(A, x, b):
    b[0] = A[0, 0] * x[0] + A[0, 1] * x[1]
    b[1] = A[1, 0] * x[0] + A[1, 1] * x[1]



@numba.jit(nopython=True)
def matrix_add_outer_product(A, c, x):
    A[0, 0] += c * x[0] * x[0]
    A[0, 1] += c * x[0] * x[1]
    A[1, 0] += c * x[1] * x[0]
    A[1, 1] += c * x[1] * x[1]



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
    'xm',
    'Sxm',
    'gamma',
    'logL',
    'entropy'
])



@numba.jit(nopython=True)
def myrand(N, state):
    state = state * 1103515245 + 12345
    return ((state//65536) % 32768) % N, state



@numba.jit(nopython=True)
def gmm_initialize_components(gmm, X, N, K):
    # initialize random state
    state = 1

    # initialize each mixture component
    for k in range(K):
        # initialize mixture weight to uniform distribution
        gmm.pi[k] = 1.0 / K

        # initialize mean to a random sample from X
        i, state = myrand(N, state)

        vector_assign(gmm.mu[k], X[i])

        # initialize covariance to identity matrix
        matrix_init_identity(gmm.sigma[k])



@numba.jit(nopython=True)
def gmm_prepare_components(gmm, K):
    D = 2

    for k in range(K):
        # compute determinant of covariance matrix
        det = matrix_determinant(gmm.sigma[k])

        # return failure if matrix inverse failed
        if det <= 0.0 or math.isnan(det):
            return False

        # compute precision matrix (inverse of covariance matrix)
        matrix_inverse(gmm.sigma[k], gmm.sigmaInv[k], det)

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
            vector_add(MP[min_k], X[i])
            counts[min_k] += 1

        # scale each mean by its sample count
        for k in range(K):
            if counts[k] > 0:
                vector_scale(MP[k], 1.0 / counts[k])

        # compute the total change of all means
        diff = 0.0

        for k in range(K):
            diff += vector_diff_norm(MP[k], gmm.mu[k])

        diff /= K
    
        # update component means
        for k in range(K):
            vector_assign(gmm.mu[k], MP[k])
        
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
            vector_assign(gmm.xm, X[i])
            vector_subtract(gmm.xm, gmm.mu[k])

            # compute Sxm = Sigma^-1 xm
            matrix_product(gmm.sigmaInv[k], gmm.xm, gmm.Sxm)

            # compute xmSxm = xm^T Sigma^-1 xm
            xmSxm = vector_dot(gmm.xm, gmm.Sxm)
            
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
        gmm.mu[k, :] = 0

        for i in range(N):
            vector_add_scaled(gmm.mu[k], gmm.gamma[i, k], X[i])

        vector_scale(gmm.mu[k], 1.0 / n_k)
        
        # update covariance matrix
        gmm.sigma[k, :, :] = 0
        
        for i in range(N):
            # compute xm = (x_i - mu_k)
            vector_assign(gmm.xm, X[i])
            vector_subtract(gmm.xm, gmm.mu[k])

            # compute Sigma_ki = gamma_ik * (x_i - mu_k) (x_i - mu_k)^T
            matrix_add_outer_product(gmm.sigma[k], gmm.gamma[i, k], gmm.xm)

        matrix_scale(gmm.sigma[k], 1.0 / n_k)



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
    
    return math.log(float(N)) * p - 2 * logL



@numba.jit(nopython=True)
def compute_icl(K, D, logL, N, E):
    p = K * (1 + D + D * D)

    return math.log(float(N)) * p - 2 * logL + 2 * E



@numba.jit(nopython=True)
def gmm_compute(
    gmm,
    x, y,
    n_samples,
    labels,
    min_samples,
    min_clusters,
    max_clusters,
    criterion):

    # perform clustering only if there are enough samples
    bestK = 0

    if n_samples >= min_samples:
        # extract clean samples from data array
        j = 0
        for i in range(len(x)):
            if labels[i] >= 0:
                gmm.data[j, 0] = x[i]
                gmm.data[j, 1] = y[i]
                j += 1

        # determine the number of clusters
        bestValue = math.inf

        for K in range(min_clusters, max_clusters + 1):
            # run the clustering model
            success = gmm_fit(gmm, gmm.data, n_samples, K, gmm.labels)

            if not success:
                continue

            # compute the criterion value of the model
            value = math.inf

            if criterion == CRITERION_AIC:
                value = compute_aic(K, 2, gmm.logL[0])
            elif criterion == CRITERION_BIC:
                value = compute_bic(K, 2, gmm.logL[0], n_samples)
            elif criterion == CRITERION_ICL:
                value = compute_icl(K, 2, gmm.logL[0], n_samples, gmm.entropy[0])

            # save the model with the lowest criterion value
            if value < bestValue:
                bestK = K
                bestValue = value
                
                # save labels for clean samples
                j = 0
                for i in range(len(x)):
                    if labels[i] >= 0:
                        labels[i] = gmm.labels[j]
                        j += 1

    return bestK



@numba.jit(nopython=True)
def pearson(x, y, labels, k, min_samples):
    n = 0
    sumx = 0.0
    sumy = 0.0
    sumx2 = 0.0
    sumy2 = 0.0
    sumxy = 0.0

    for i in range(len(x)):
        if labels[i] == k:
            x_i = x[i]
            y_i = y[i]

            sumx += x_i
            sumy += y_i
            sumx2 += x_i * x_i
            sumy2 += y_i * y_i
            sumxy += x_i * y_i
            
            n += 1

    if n >= min_samples:
        return (n*sumxy - sumx*sumy) / math.sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy))

    return np.nan



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
def spearman(x, y, labels, k, min_samples, x_rank, y_rank):
    # extract samples in pairwise cluster
    n = 0

    for i in range(len(x)):
        if labels[i] == k:
            x_rank[n] = x[i]
            y_rank[n] = y[i]
            n += 1

    # get power of 2 size
    for i in range(n, len(x_rank)):
        x_rank[i] = math.inf
        y_rank[i] = math.inf

    # compute correlation only if there are enough samples
    if n >= min_samples:
        # compute rank of x
        bitonic_sort_ff(x_rank, y_rank)
        compute_rank(x_rank)

        # compute rank of y
        bitonic_sort_ff(y_rank, x_rank)
        compute_rank(y_rank)

        # compute correlation of rank arrays
        sumx = 0
        sumy = 0
        sumx2 = 0
        sumy2 = 0
        sumxy = 0

        for i in range(n):
            x_i = x_rank[i]
            y_i = y_rank[i]

            sumx += x_i
            sumy += y_i
            sumx2 += x_i * x_i
            sumy2 += y_i * y_i
            sumxy += x_i * y_i

        return (n*sumxy - sumx*sumy) / math.sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy))

    return np.nan



@numba.jit(nopython=True)
def similarity_kernel(
    x, y,
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
    x_sorted,
    y_sorted,
    gmm,
    labels,
    correlations):

    # fetch pairwise input data
    n_samples = fetch_pair(
        x, y,
        minexpr,
        maxexpr,
        labels)

    # remove pre-clustering outliers
    if preout:
        n_samples = mark_outliers(
            x, y,
            labels,
            0,
            -7,
            x_sorted,
            y_sorted)

    # perform clustering
    K = 1

    if clusmethod == CLUSMETHOD_GMM:
        K = gmm_compute(
            gmm,
            x, y,
            n_samples,
            labels,
            minsamp,
            minclus,
            maxclus,
            criterion)

    # remove post-clustering outliers
    if K > 1 and postout:
        for k in range(K):
            n_samples = mark_outliers(
                x, y,
                labels,
                k,
                -8,
                x_sorted,
                y_sorted)

    # perform correlation
    if corrmethod == CORRMETHOD_PEARSON:
        for k in range(K):
            correlations[k] = pearson(
                x, y,
                labels,
                k,
                minsamp)

    elif corrmethod == CORRMETHOD_SPEARMAN:
        for k in range(K):
            correlations[k] = spearman(
                x, y,
                labels,
                k,
                minsamp,
                x_sorted,
                y_sorted)

    # save number of clusters
    return K



def write_pair(
    i, j,
    K,
    labels,
    correlations,
    mincorr,
    maxcorr,
    outfile):

    # determine number of valid correlations
    valid = np.array([(~np.isnan(corr) and mincorr <= abs(corr) and abs(corr) <= maxcorr) for corr in correlations])
    n_clusters = valid.sum()
    cluster_idx = 1

    # write each correlation to output file
    for k in range(K):
        corr = correlations[k]

        # make sure correlation meets thresholds
        if valid[k]:
            # compute sample mask
            y_k = np.copy(labels)
            y_k[(y_k >= 0) & (y_k != k)] = 0
            y_k[y_k == k] = 1
            y_k[y_k < 0] *= -1

            sample_mask = ''.join([str(y_i) for y_i in y_k])

            # compute summary statistics
            n_samples = (y_k == 1).sum()

            # write correlation to output file
            outfile.write('%d\t%d\t%d\t%d\t%d\t%0.8f\t%s\n' % (i, j, cluster_idx, n_clusters, n_samples, corr, sample_mask))

            # increment cluster index
            cluster_idx += 1



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
    maxcorr,
    outfile):

    # initialize workspace
    N = emx.shape[1]
    N_pow2 = next_power_2(N)
    K = maxclus

    x_sorted = np.empty((N_pow2,), dtype=np.float32)
    y_sorted = np.empty((N_pow2,), dtype=np.float32)

    gmm = GMM(
        data       = np.empty((N, 2), dtype=np.float32),
        labels     = np.empty((N,), dtype=np.int8),
        pi         = np.empty((K,), dtype=np.float32),
        mu         = np.empty((K, 2), dtype=np.float32),
        sigma      = np.empty((K, 2, 2), dtype=np.float32),
        sigmaInv   = np.empty((K, 2, 2), dtype=np.float32),
        normalizer = np.empty((K,), dtype=np.float32),
        MP         = np.empty((K, 2), dtype=np.float32),
        counts     = np.empty((K,), dtype=np.int32),
        logpi      = np.empty((K,), dtype=np.float32),
        xm         = np.empty((2,), dtype=np.float32),
        Sxm        = np.empty((2,), dtype=np.float32),
        gamma      = np.empty((N, K), dtype=np.float32),
        logL       = np.empty((1,), dtype=np.float32),
        entropy    = np.empty((1,), dtype=np.float32)
    )

    labels = np.empty((N,), dtype=np.int8)
    correlations = np.empty((K,), dtype=np.float32)

    # process each gene pair
    for i in range(emx.shape[0]):
        # print("%8d" % (i))

        for j in range(i):
            # extract pairwise data
            x = emx[i]
            y = emx[j]

            # compute pairwise similarity
            K = similarity_kernel(
                x, y,
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
                x_sorted,
                y_sorted,
                gmm,
                labels,
                correlations)

            # save pairwise results
            write_pair(
                i, j,
                K,
                labels,
                correlations,
                mincorr,
                maxcorr,
                outfile)


                    
@cuda.jit
def similarity_gpu_helper(
    n_pairs,
    in_emx,
    in_index,
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
    work_x,
    work_y,
    work_gmm_data,
    work_gmm_labels,
    work_gmm_pi,
    work_gmm_mu,
    work_gmm_sigma,
    work_gmm_sigmaInv,
    work_gmm_normalizer,
    work_gmm_MP,
    work_gmm_counts,
    work_gmm_logpi,
    work_gmm_xm,
    work_gmm_Sxm,
    work_gmm_gamma,
    work_gmm_logL,
    work_gmm_entropy,
    out_K,
    out_labels,
    out_correlations):

    # get global index
    i = cuda.grid(1)
    
    if i >= n_pairs:
        return
    
    # initialize workspace variables
    index_x, index_y = in_index[i]
    x = in_emx[index_x]
    y = in_emx[index_y]
    x_sorted = work_x[index_x]
    y_sorted = work_y[index_y]

    gmm = GMM(
        data       = work_gmm_data[i],
        labels     = work_gmm_labels[i],
        pi         = work_gmm_pi[i],
        mu         = work_gmm_mu[i],
        sigma      = work_gmm_sigma[i],
        sigmaInv   = work_gmm_sigmaInv[i],
        normalizer = work_gmm_normalizer[i],
        MP         = work_gmm_MP[i],
        counts     = work_gmm_counts[i],
        logpi      = work_gmm_logpi[i],
        xm         = work_gmm_xm[i],
        Sxm        = work_gmm_Sxm[i],
        gamma      = work_gmm_gamma[i],
        logL       = work_gmm_logL[i],
        entropy    = work_gmm_entropy[i]
    )

    labels = out_labels[i]
    correlations = out_correlations[i]

    # save number of clusters
    out_K[i] = similarity_kernel(
        x, y,
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
        x_sorted,
        y_sorted,
        gmm,
        labels,
        correlations)



def pairwise_increment(i, j):
    j += 1
    if i == j:
        i += 1
        j = 0
    return i, j



def similarity_gpu(
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
    maxcorr,
    gsize,
    lsize,
    outfile):

    # allocate device buffers
    W = gsize
    N = emx.shape[1]
    N_pow2 = next_power_2(N)
    K = maxclus

    in_emx               = cuda.to_device(emx)
    in_index_cpu         = cuda.pinned_array((W, 2), dtype=np.int32)
    in_index_gpu         = cuda.device_array_like(in_index_cpu)
    work_x               = cuda.device_array((W, N_pow2), dtype=np.float32)
    work_y               = cuda.device_array((W, N_pow2), dtype=np.float32)
    work_gmm_data        = cuda.device_array((W, N, 2), dtype=np.float32)
    work_gmm_labels      = cuda.device_array((W, N), dtype=np.int8)
    work_gmm_pi          = cuda.device_array((W, K), dtype=np.float32)
    work_gmm_mu          = cuda.device_array((W, K, 2), dtype=np.float32)
    work_gmm_sigma       = cuda.device_array((W, K, 2, 2), dtype=np.float32)
    work_gmm_sigmaInv    = cuda.device_array((W, K, 2, 2), dtype=np.float32)
    work_gmm_normalizer  = cuda.device_array((W, K), dtype=np.float32)
    work_gmm_MP          = cuda.device_array((W, K, 2), dtype=np.float32)
    work_gmm_counts      = cuda.device_array((W, K), dtype=np.int32)
    work_gmm_logpi       = cuda.device_array((W, K), dtype=np.float32)
    work_gmm_xm          = cuda.device_array((W, 2), dtype=np.float32)
    work_gmm_Sxm         = cuda.device_array((W, 2), dtype=np.float32)
    work_gmm_gamma       = cuda.device_array((W, N, K), dtype=np.float32)
    work_gmm_logL        = cuda.device_array((W, 1), dtype=np.float32)
    work_gmm_entropy     = cuda.device_array((W, 1), dtype=np.float32)
    out_K_cpu            = cuda.pinned_array((W,), dtype=np.int8)
    out_K_gpu            = cuda.device_array_like(out_K_cpu)
    out_labels_cpu       = cuda.pinned_array((W, N), dtype=np.int8)
    out_labels_gpu       = cuda.device_array_like(out_labels_cpu)
    out_correlations_cpu = cuda.pinned_array((W, K), dtype=np.float32)
    out_correlations_gpu = cuda.device_array_like(out_correlations_cpu)

    # iterate through global work blocks
    n_genes = emx.shape[0]
    n_total_pairs = n_genes * (n_genes - 1) // 2

    index_x = 1
    index_y = 0

    for i in range(0, n_total_pairs, gsize):
        # print("%8d %8d" % (i, n_total_pairs))

        # determine number of pairs
        n_pairs = min(gsize, n_total_pairs - i)

        # initialize index array
        index_x_ = index_x
        index_y_ = index_y

        for j in range(n_pairs):
            in_index_cpu[j] = index_x_, index_y_
            index_x_, index_y_ = pairwise_increment(index_x_, index_y_)

        # copy index array to device
        in_index_gpu.copy_to_device(in_index_cpu)

        # execute similarity kernel
        similarity_gpu_helper[gsize // lsize, lsize](
            n_pairs,
            in_emx,
            in_index_gpu,
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
            work_x,
            work_y,
            work_gmm_data,
            work_gmm_labels,
            work_gmm_pi,
            work_gmm_mu,
            work_gmm_sigma,
            work_gmm_sigmaInv,
            work_gmm_normalizer,
            work_gmm_MP,
            work_gmm_counts,
            work_gmm_logpi,
            work_gmm_xm,
            work_gmm_Sxm,
            work_gmm_gamma,
            work_gmm_logL,
            work_gmm_entropy,
            out_K_gpu,
            out_labels_gpu,
            out_correlations_gpu
        )
        cuda.synchronize()

        # copy results from device
        out_K_gpu.copy_to_host(out_K_cpu)
        out_labels_gpu.copy_to_host(out_labels_cpu)
        out_correlations_gpu.copy_to_host(out_correlations_cpu)

        # save correlation matrix to output file
        index_x_ = index_x
        index_y_ = index_y

        for j in range(n_pairs):
            # extract pairwise results
            K = out_K_cpu[j]
            labels = out_labels_cpu[j]
            correlations = out_correlations_cpu[j, 0:K]

            # save pairwise results
            write_pair(
                index_x_, index_y_,
                K,
                labels,
                correlations,
                mincorr,
                maxcorr,
                outfile)

            # increment pairwise index
            index_x_, index_y_ = pairwise_increment(index_x_, index_y_)

        # update local pairwise index
        index_x = index_x_
        index_y = index_y_



def main():
    if len(sys.argv) != 4:
        print("usage: ./kinc-numba.py <infile> <outfile> <gpu>")
        sys.exit(-1)

    # define input parameters
    args_input = sys.argv[1]
    args_output = sys.argv[2]
    args_gpu = bool(int(sys.argv[3]))
    args_clusmethod = CLUSMETHOD_GMM
    args_corrmethod = CORRMETHOD_SPEARMAN
    args_preout = True
    args_postout = True
    args_minexpr = 0.0
    args_maxexpr = 20.0
    args_minsamp = 30
    args_minclus = 1
    args_maxclus = 5
    args_criterion = CRITERION_ICL
    args_mincorr = 0.5
    args_maxcorr = 1.0
    args_gsize = 4096
    args_lsize = 32

    # load input data
    emx = pd.read_csv(args_input, sep='\t', index_col=0)
    emx_data = emx.values.astype(np.float32)

    # initialize output file
    outfile = open(args_output, 'w')

    # run similarity
    if args_gpu:
        similarity_gpu(
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
            args_maxcorr,
            args_gsize,
            args_lsize,
            outfile
        )

    else:
        similarity_cpu(
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
            args_maxcorr,
            outfile
        )



if __name__ == '__main__':
    main()
