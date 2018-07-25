import numpy as np

np.random.seed()

def f():
    return np.random.uniform(0,1)

def main():
    x = f()
    print x

if __name__ == '__main__':
    main()
