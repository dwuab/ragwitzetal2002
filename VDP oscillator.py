import pyximport
pyximport.install()
from VDP import VDP_integrate, embeddingPrediction
from matplotlib.pyplot import figure, show, plot, legend, xlabel, ylabel,scatter, savefig
import matplotlib.pyplot as plt
from numpy import array, random, std, arange, copy
from math import sqrt
import sys
import numpy as np
from statsmodels.tsa.stattools import acf
import time


def test_embedding(a, b, X0=2.0, Y0=2.0, eps=0.1, testLength = 1000, n_neigh=4, predict_y=False):
    '''
    equation:
    Vx(t) = y(t)+aG(t)
    Vy(t) = [r-x(t)^2]*y(t)-x(t)+bG(t)
    '''

    '''tunable variables for van der pol oscillator'''
    r = 3.0
    T = 15 #time delay in terms of h
    startTime = 0.  # start of interval
    endTime = 100000  # end of interval
    samplingRate = 100  # number of steps in 1 second
    N = int(samplingRate * (endTime - startTime))  # number of steps
    h = 1. / samplingRate  # size of single step

    x_ts = np.zeros([N+1])
    y_ts = np.zeros([N+1])

    VDP_start_t = time.time()
    VDP_integrate(a, b, r, X0, Y0, endTime, samplingRate, x_ts, y_ts, 10.)
    VDP_elapsed_t = time.time() - VDP_start_t
    print("VDP integration elapsed time: ", VDP_elapsed_t)

    "tunable parameters for embedding predictors"
    tau_min = 1 #delay T from embTMin, embTMin+tau_inc, embTMin+2embTChange, ... , tau_max
    tau_max = 15
    tau_inc = 1
    m_min = 2 #dimensions in condition vector starts from m_min, m_min+m_inc, m_min+2mChange, ..., m_max
    m_max = 4
    m_inc = 1
    "fixed parameters for embedding predictors"
    predictStart = N - testLength + 1 #predictions from array[predictStart] to array[N]
    predictEnd = N

    t_ts = np.linspace(0., endTime, N+1)

    # sample_acf = acf(x_ts, nlags=500)
    # act = 0
    # while sample_acf[act] > np.exp(-1):
    #     act += 1
    # print("autocorrelation time: ", h*act)

    # figure()
    # plot(t_ts, x_ts, label='x')
    # plot(t_ts, y_ts, label='y')
    # xlabel('t')
    # ylabel('d')
    # legend()
    # savefig('tdgraph '+" eps_"+str(eps)+' neighbor_'+str(n_neigh)+" a_"+str(a)+" b_"+str(b)+".png", format='png')
    #
    # figure()
    # #plot(x_ts[:int(N-T*secWidth+1)], x_ts[int(T*secWidth):])
    # plot(x_ts[:int(N-T+1)], x_ts[int(T):])
    # #scatter(xPlot[:int((endTime-startTime)-T+1)], xPlot[int(T):])
    # xlabel('x(t)')
    # ylabel('x(t-' + str(T*h)+')')
    # savefig('Xembed '+" eps_"+str(eps)+' neighbor_'+str(n_neigh)+" a_"+str(a)+" b_"+str(b)+".png", format='png')
    #
    #
    # figure()
    # #plot(y_ts[:int(N-T*secWidth+1)], y_ts[int(T*secWidth):])
    # plot(y_ts[:int(N-T+1)], y_ts[int(T):])
    # xlabel('y(t)')
    # ylabel('y(t-' + str(T*h)+')')
    # savefig('Yembed '+" eps_"+str(eps)+' neighbor_'+str(n_neigh)+" a_"+str(a)+" b_"+str(b)+".png", format='png')

    '''prediction part'''
    if predict_y:
        actualTimeSeries=y_ts[predictStart:predictEnd+1]
    else:
        actualTimeSeries=x_ts[predictStart:predictEnd+1]
    sdTest = std(actualTimeSeries)
    errorDelayList = np.zeros([m_max-m_min+1, tau_max-tau_min+1])
    for m_idx, m in enumerate(range(m_min, m_max+1, m_inc)):
        for tau_idx, tau in enumerate(range(tau_min, tau_max+1, tau_inc)):
            try:
                predTimeSeries=[]
                error=0
                for t in range(predictStart,predictEnd+1):
                    if predict_y:
                        predTimeSeries.append(embeddingPrediction(m, tau, y_ts, t, n_neigh, eps))
                    else:
                        predTimeSeries.append(embeddingPrediction(m, tau, x_ts, t, n_neigh, eps))
                # for t in range(testLength):
                #     error += (predTimeSeries[t]-actualTimeSeries[t])**2
                error = sum((predTimeSeries-actualTimeSeries)**2)
                error /= testLength
                error = sqrt(error)
                error /= sdTest
            except:
                error = 0.
            errorDelayList[m_idx,tau_idx] = error
            print(m,tau,error)
        plot(range(tau_min, tau_max+1, tau_inc), errorDelayList[m_idx, :], 'o-', label=r'$m$='+str(m))
        # figure(5)
        # plot(range(1,testLength+1),predTimeSeries-actualTimeSeries, label='m='+str(m))
        # print(m)
    xlabel(r'Delay $\tau$', fontsize=20)
    ylabel(r'relative prediction error', fontsize=16)
    legend(loc='best', fontsize=16)
    min_error = np.min(errorDelayList)
    max_error = np.max(errorDelayList)
    gap_error = max_error - min_error
    plt.ylim([min_error, min_error+gap_error*0.3])
    if predict_y:
        savefig('predict_error_predict_len_{} eps_{} nei_{} a_{} b_{}_y.pdf'.format(testLength, eps, n_neigh, a, b))
    else:
        savefig('predict_error_predict_len_{} eps_{} nei_{} a_{} b_{}_x.pdf'.format(testLength, eps, n_neigh, a, b))
    plt.close()

if __name__=="__main__":
    for n_neigh in range(5, 51, 5):
        for eps in np.linspace(0.6, 1.0, 5):
            test_embedding(a=0.5, b=0.0, testLength=5000, eps=eps, n_neigh=n_neigh, predict_y=True)
