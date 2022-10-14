import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import multiprocessing

#  Lorentz System
def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result



def alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(alg2(data[:split]))
        right = iter(alg2(data[split:]))
        result = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top:
                result.append(left_top)
                try:
                    left_top = next(left)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right)
            else:
                result.append(right_top)
                try:
                    right_top = next(right)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left)

def merge(left, right):
    left = iter(left)
    right = iter(right)
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(left)
    right_top = next(right)
    while True:
        if left_top < right_top:
            result.append(left_top)
            try:
                left_top = next(left)
            except StopIteration:
                # nothing remains on the left; add the right + return
                return result + [right_top] + list(right)
        else:
            result.append(right_top)
            try:
                right_top = next(right)
            except StopIteration:
                # nothing remains on the right; add the left + return
                return result + [left_top] + list(left)


#%%
if __name__ == '__main__':
    
    n_lst = [10, 1000, 10000, 100000, 1000000, 10000000]
    time1 = []
    time2 = []

    for n in n_lst:
        print('n = ' + str(n))
        data = data1(n)
        
        t_start = perf_counter()

        with multiprocessing.Pool(2) as pool:
            sor = pool.map(alg2, [data[:len(data)//2],data[len(data)//2:]])
            merge(sor[0],sor[1])
        t_stop = perf_counter()
        print('multi_threading alg2')
        print(t_stop-t_start)
        print('')
        time1.append(t_stop-t_start)
        
        t_start = perf_counter()
        ret2 = alg2(data)
        t_stop = perf_counter()
        print('original alg2')
        print(t_stop-t_start)
        print('')
        time2.append(t_stop-t_start)


    plt.plot(n_lst, time1, label='multiprocessing')
    plt.plot(n_lst, time2, label='original')
    plt.legend()
    plt.title('data 1')
    plt.show()
