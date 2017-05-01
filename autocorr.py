import numpy as np
def calculate_autocorrelation(arr):
    mean = np.mean(arr)
    y = np.correlate(arr-mean,arr-mean,'same')
    L = len(y)
    ceily = int(np.ceil(len(y)/2))
    C_t = y[ceily:]
    print C_t
    C_t = C_t/(L-1-np.arange(ceily))
    C_t = C_t/C_t[0]
    return C_t
    
def calculate_autocorrelation_time(arr,w=5):
    L = arr.shape[0]
    M = 1
    tau = 1.0
    while(True):
        if (tau>1.0 and M > w*tau):
            print M
            return tau

        if (w * tau) >= L:
            print "Series too short!"
            return tau
        M = M + 1
        tau = tau + 2 * arr[M]
