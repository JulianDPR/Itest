# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:46:39 2024

@author: Julian
"""

from tools import Cn
import numpy as np
import scipy.stats as ss
import time as t
import pandas as pd
import multiprocessing as mtp
from tools import cn_gen

#%%


if __name__ == "__main__":
    
    a = t.time()
    
    
    for i in range(8):
        
        cn_gen(1000)
    
    
    b = t.time()
    
    print(b - a)
    
    
    manager = mtp.Manager()
    
    return_dict = manager.dict()
    
    p1 = mtp.Process(target=cn_gen, args = (1000,))
    p2 = mtp.Process(target=cn_gen, args = (1000,))
    p3 = mtp.Process(target=cn_gen, args = (1000,))
    p4 = mtp.Process(target=cn_gen, args = (1000,))
    p5 = mtp.Process(target=cn_gen, args = (1000,))
    p6 = mtp.Process(target=cn_gen, args = (1000,))
    p7 = mtp.Process(target=cn_gen, args = (1000,))
    p8 = mtp.Process(target=cn_gen, args = (1000,))

    a = t.time()

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    b = t.time()
    
    print(b-a)


