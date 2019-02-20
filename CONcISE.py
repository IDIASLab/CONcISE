from __future__ import division
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import os
import matplotlib.pyplot as plt
################################## input #####################################
# input:
# cn: costs for using features
# C: list of costs for features
# n: number of features
# L: number of decisions
# w: [0:0.01:1]
# features
# labels
# output:
# g(w)
# V(w)
# Vtilde(w)
### this function computes the probabilities of some data point being Bullying and Normal respectively, observing n-th feature being value fn
def prob_dist(fn,V_f,labels):
    # input
    # fn: numerical value of nth-feature for some message
    # V_f: array of feature
    # labels: array of labels, B or N
    # type : pdf or pmf
    dist_B = np.array(V_f[labels == 'B'])
    dist_N = np.array(V_f[labels == 'N'])
    if fn == int(fn):

            P_B = list(dist_B).count(fn)/len(dist_B)
            P_N = list(dist_N).count(fn)/len(dist_N)
    else: # if continuous, use cdf
        X_B = np.array(V_f[labels == 'B']) # n-th feature vector of bullies
        X_N = np.array(V_f[labels == 'N'])  # n-th feature vector of normal
        ecdf_B = ECDF(X_B)
        ecdf_N =  ECDF(X_N)
        X_plot = np.linspace(min(V_f), max(V_f), 300)#[:, np.newaxis]
        arr = np.abs(X_plot - fn)
        [floor,ceiling] = sorted(np.argpartition(arr, 2)[:2])# find the small interval that fn lies in
        P_B = (ecdf_B(X_plot[ceiling]) - ecdf_B(X_plot[floor])) / (ceiling - floor)
        P_N = ecdf_N(X_plot[ceiling]) - ecdf_N(X_plot[floor]) / (ceiling - floor)

    return P_B,P_N

def Pi_n(p0,n,test, data,labels):
    # n (>0) is the number of features
    a = p0
    b = 1-p0
    for i in range(n):
        P = prob_dist(test[i], data[columns[i]], labels)
        P_B = P[0]
        P_N = P[1]
        if P_B * P_N != 0:
            a = a * P_B
            b = b * P_N

    Pi_n = a / (a + b)
    return Pi_n

p0 = 0.3
train = pd.read_csv('train.csv')
K = 10 # number of features in the dataset
columns = train.columns.tolist()[:K]


def g(w):
    g = []
    for j in range(L):
        a = np.multiply(C[0][j], w)
        b = np.multiply(C[1][j], 1 - w)
        g_j = a + b
        g.append(g_j)
    g_w = np.amin(g, axis=0)
    return g_w

if not os.path.exists('offline/'):
    os.makedirs('offline/')
if not os.path.exists('results/'):
    os.makedirs('results/')

cv = [0.003,0.002,0.0015,0.0013]
for c in cv[:1]:
    avg_folds = 0  # average number of features used in cv
    num_f_used = []  # list of the number of features used
    K = len(columns)
    D = [1,2]
    L = len(D)
    l = ['matched', 'nonmatched']
    C = np.empty([2, L])
    C[0][0]= 0
    C[0][1] = 1
    C[1][0] = 1
    C[1][1] = 0
    cn = np.full(len(columns), c)
    f_names= np.array(columns)
    n_folder = 1 # number of folders for cross validation
    data = train
    step = 1000
    w =np.linspace(0,1,step)
    g_w = g(w)
    Vn = np.zeros(shape=(K+1,step))
    Vntilde = np.zeros(shape=(K + 1, step))
    Vn[K] = g_w
    df = [] # initialize dataframe
    for i in range(K-1,-1,-1):
        Sigma= np.zeros(step)
        labels = data['labels']
        labels = np.array(['B' if x == 'T' else 'N' for x in labels])
        V_f = data[columns[i]]
        support = np.unique(V_f)
        if np.issubdtype(support[0], float):
            support = np.linspace(np.amin(support),np.amax(support),100)
        for j in range(len(support)):
            P_B = prob_dist(support[j], V_f, labels)[0]
            P_N = prob_dist(support[j],V_f,labels)[1]
            if P_N*P_B==0:
                P_B=P_N=1
            else:
                a = np.add(np.multiply(w, P_B),np.multiply(np.ones(step) - w, P_N))
                b = np.divide(np.multiply(w, P_B), np.add(np.multiply(w, P_B), np.multiply(np.ones(step) - w, P_N)))
                index = np.array(b*step-1)
                I = []
                for ind in index:
                    if np.isnan(ind):
                        ind = 0
                    I.append(int(ind))
                Sigma = Sigma + np.multiply(a, Vn[i + 1][I])
            Vntilde[i] = np.add(cn[i], Sigma)
            Vn[i] = np.minimum(g_w, Vntilde[i])
    df = pd.DataFrame(Vntilde)
    df.to_csv('offline/Vntilde_'+str(c)+'.csv',index=False)
    #
    # """uncomment below to see the plots of g(w) and V~"""
    # plt.plot(w, g(w))
    # for i in range(K):
    #     plt.plot(w,g(w),'k--')
    #     plt.plot(w, Vntilde[i], label=columns[i])
    #     legend = plt.legend(shadow=True)
    #     plt.savefig('offline/Vntilde_plot_'+str(c)+'.png')
    # plt.close()

######################################### sequential part ##################################
    Vntilde = np.array(pd.read_csv('offline/Vntilde_'+str(c)+'.csv'))
    posterior = []
    R = []
    T = pd.read_csv('test.csv', usecols=columns)
    data_test = pd.read_csv('test.csv')
    idx_s = range(204, 207)
    avg = 0  # initialize average number of features used in each fold
    results = [1, 0]
    prediction = []

    La = []
    S_df = pd.DataFrame()
    used_msg = []
    S_label = []
    pi_B = []
    for jj in range(len(idx_s)):
        T_s = []  # timer for each session

        num_f = []
        y_score = []

        i_s = idx_s[jj]
        test = T[data_test['idx'] == i_s]
        z = 0
        alarm = 0
        while z < len(test):
            r = 0
            intertime = 0
            while r < K:
                pi_n = Pi_n(p0, r + 1, np.array(test[columns])[z], data, labels)
                pi_approx_ind = np.abs(w - pi_n).argmin()
                condition = g(w[pi_approx_ind]) - Vntilde[r][pi_approx_ind]
                r = r + 1
                if condition <= 0 or r == len(
                        columns):  # for sequential. uncomment the next line to output non-sequential
                    # if r == len(columns):
                    break

            z = z + 1
            num_f.append(r)
            D = []
            for j in range(L):
                D.append(C[0][j] * pi_n + C[1][j] * (1 - pi_n))
            D0 = D.index(min(D))
            y_score.append(pi_n)
            if results[D0] == 1:
                alarm = alarm + 1
            if alarm == 5:
                S_label.append(1)
                used_msg.append(z)
                R.append(np.mean(num_f))
                pi_B.append(np.mean([x for x in y_score if x > 0.5]))
                break
            if z == len(test) and alarm < 5:
                used_msg.append(z)
                S_label.append(0)
                R.append(np.mean(num_f))
                pi_B.append(np.mean(y_score))

    S_df['pred'] = S_label
    S_df['# msg'] = used_msg
    S_df['avg_f'] = R
    S_df['pi_B'] = pi_B
    S_df.to_csv('results/c='+str(c)+'.csv')



