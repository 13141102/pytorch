import multiprocessing as mp

def job(q, a, b):
    res = 0
    for i in range(1000):
        res += i + i**2 + i**3
        q.put(res) #queue

if __name__ == '__main__':
    q = mp.Queue()