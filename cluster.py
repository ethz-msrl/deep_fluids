import os
import sys
import multiprocessing
from subprocess import call

args = None

def mp_workder(args):    
    print(args)
    call(args)

if __name__ == '__main__':
    num_args, args = len(sys.argv)-1, list(sys.argv)[1:]
    for i in range(num_args):
        args[i] = ['python', 'main.py'] + args[i].split() + ['--gpu_id=%d' % i]
    p = multiprocessing.Pool(num_args)
    p.map(mp_workder, args)
