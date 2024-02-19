import numpy as np
import pandas as pd
import scipy.stats as ss
import time as t
from tools import ranks
import matplotlib.pyplot as plt

def Cn_(XY, u, v):
    n = len(XY)
    R = np.argsort(XY[:, 0]) + 1
    S = np.argsort(XY[:, 1]) + 1
    
    count = np.sum((R / (n + 1) <= u[:, None]) & (S / (n + 1) <= v[:, None]), axis=0)
    
    cn = count / n
    
    return cn

def Cn(XY, u, v):
    
    n, R, S = ranks(XY)
    cn = np.sum((R / (n + 1) <= u[:, None]) & (S / (n + 1) <= v[:, None]), axis=1)/n
    
    return cn

def Cn__(XY, u, v):
    
    n, R, S = ranks(XY)
    cn = ((R / (n + 1) <= u) * (S / (n + 1) <= v)).sum()/n
    
    return cn



n = 1000
u = np.array([0.97, 0.54, 0.73, 0.17, 0.32])
v = np.array([0.21, 0.18, 0.55, 0.36, 0.82])
XY = pd.DataFrame(dict(x=u, y=v)).to_numpy()
u,v = np.meshgrid(*[np.linspace(1/5, 1, 5),np.linspace(1/5, 1, 5)])

print(np.matrix(list(map(lambda x,y: Cn(pd.DataFrame(XY, columns = ["x","y"]),x, y), u, v))))

print(np.matrix(list(map(lambda x,y: np.vectorize(Cn__, excluded=[0])(pd.DataFrame(XY, columns = ["x","y"]),x,y), u, v))))


