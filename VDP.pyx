from libc.math cimport sqrt, abs
import numpy as np
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cpdef int VDP_integrate(double a, double b, double r, double X0, double Y0, int endTime, int secWidth, \
double[::1] xSmall, double[::1] ySmall, double transientTime) except *:
    cdef:
        double h
        double startTime = 0.
        double x, y, xNew, yNew
        double[::1] scaled_noise
        int t, N

    "fixed parameters for van der pol oscillator"
    N = secWidth * <int>(endTime - startTime)  # number of steps
    trasient_N = secWidth * <int>(transientTime)
    h = (endTime - startTime) / N  # size of single step
    scaled_noise = sqrt(h) * np.random.normal(0, 1, N+trasient_N)  # generate gaussian noise with mean=0 and sd=1 with length = N
    x = X0
    y = Y0
    xSmall[0] = X0
    ySmall[0] = Y0

    for t in range(trasient_N):
        xNew = x + y*h + a*scaled_noise[t]
        yNew = y + (r-x**2)*y*h - x*h + b*scaled_noise[t]
        x = xNew
        y = yNew

    xSmall[0] = x
    ySmall[0] = y

    '''start simulation'''
    for t in range(1, N+1):
        xNew = x + y*h + a*scaled_noise[t-1+trasient_N]
        yNew = y + (r-x**2)*y*h - x*h + b*scaled_noise[t-1+trasient_N]
        x = xNew
        y = yNew
        xSmall[t] = x
        ySmall[t] = y

@boundscheck(False)
@wraparound(False)
cpdef double embeddingPrediction(int d, int tau, double[::1] arr, int tt, int nuOfNeigh, double eps) except *:
    cdef:
        double[:] condVector = arr[tt-1:tt-1-tau * d:-tau]
        int countNeigh = 0
        double predNeigh = 0.
        int back = 2, i
        double[:] neighVector
        double distance2
    while countNeigh < nuOfNeigh:
        if (tt - back - (d-1) * tau)<0:
            raise Exception('error\nexceeds time series length when sampling')
        neighVector = arr[tt-back:tt-back-tau*d:-tau]

        distance2 = abs(neighVector[0] - condVector[0])

        # for i in range(d):
        #     distance2 += (neighVector[i] - condVector[i])**2
        # distance2 = sqrt(distance2)

        for i in range(1, d):
            if abs(neighVector[i] - condVector[i]) > distance2:
                distance2 = abs(neighVector[i] - condVector[i])

        if distance2 <= eps:
            countNeigh += 1
            predNeigh += arr[tt - back + 1]
            # print(distance2, back)
        back += 1

    predNeigh /= nuOfNeigh
    return predNeigh