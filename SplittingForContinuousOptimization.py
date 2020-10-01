from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import time


def peaks(x):
    """ Example function to minimise. """
    return 3 * (1 - x[0])**2 * np.exp(-(x[0]**2) - (x[1] + 1)**2) \
           - 10 * (x[0] / 5 - x[0] ** 3 - x[1]**5) * np.exp(-x[0]**2 - x[1]**2) \
           - 1/3 * np.exp(-(x[0] + 1)**2 - x[1]**2) 


def SCO(S, N, ς, w, B, MaxTry=5):
    """
    Implementation of the Splitting for Continuous Optimization Algorithm.
    This algorithm will attempt to minimise a function with multidimensional inputs.
    For further details, see Data Science and Machine Learning (Algorithm 3.4.4).
    
    Parameters
    ----------
        S: (function) Objective function.
        N: (int) Sample size.
        ς: (float) Rarity parameter.
        w: (float) Scale factor.
        B: (list) Upper and lower bounds for a global minimiser.
        MaxTry: (int) Maximum number of attempts.
    
    Returns
    -------
        Sequence of the best (argmin(S(X)), min(S(X))) in each iteration.    
    """
    time_start = time.time()
    
    Y_SET = {0: [np.random.uniform(B[0], B[1]) for _ in range(N)]}
    X_SET = {}
    X_best = {}
    b = {}
    t = 0
    N_elite = int(np.ceil(N * ς))
    
    n = len(B[0])
    
    R = np.zeros(N_elite) 
    σ = {}              
    I_SET = np.arange(N_elite)
    
    B_i = np.concatenate((np.ones(N - N_elite), np.zeros(2*N_elite - N)))
    
    best = np.inf
    while True:
        print(f"ITERATION {t+1}...")
        S_X = np.array([S(X) for X in Y_SET[t]])
        idx = np.argpartition(S_X, N_elite)[:N_elite]
        S_X = S_X[idx]
        X = [Y_SET[t][i] for i in idx]
        X_SET[t+1] = X
        b[t+1] = S_X[0]
        X_best[t+1] = X[0].copy()
        
        np.random.shuffle(B_i)
        
        for i in range(N_elite):
            R[i] = int(np.floor(N / N_elite) + B_i[i])  # random splitting factor
            Y = X[i].copy()
            Y_dash = Y.copy()
            
            for j in range(int(R[i])):
                I = np.random.choice(I_SET[I_SET != i])
                σ[i] = w * np.abs(X[i] - X[I])
                π = np.random.permutation(n)
                
                for k in range(n):
                    for Try in range(MaxTry):
                        Z = np.random.normal()
                        Y_dash[π[k]] = Y[π[k]] + σ[i][π[k]] * Z
                        if S(Y_dash) < S(Y):
                            Y = Y_dash.copy()
                            break
                        
                if Y_SET.get(t+1) == None:
                    Y_SET[t+1] = []
                Y_SET[t+1].append(Y.copy())
                
        t = t + 1
        if b[t] < best:
            best = b[t]
            print(f"Best value: {best}")
        
        # stopping condition
        STOP = t > 1 and abs(b[t-1] - b[t]) < 10**(-4) or t >= 10**3
        if STOP:
            time_stop = time.time()
            print(f"\nTerminated after {time_stop - time_start} seconds.")
            break
        
    return (X_best, b)


def main():
    # Plot the function to minimise
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1 = np.linspace(-3, 3, 1000)
    x2 = np.linspace(-3, 3, 1000)
    x1, x2 = np.meshgrid(x1, x2)
    y = peaks(np.array([x1, x2]))
    ax.plot_surface(x1, x2, y, cmap='viridis', linewidth=0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    plt.show()
    
    # Minimise the function
    SCO(S=peaks, N=200, ς=0.8, w=0.5, B=[-3*np.ones(2), 3*np.ones(2)])
    
    
if __name__ == "__main__":
    main()
    