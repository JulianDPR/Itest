# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:46:00 2024

@author: Julian
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#%% Gumbel-Barnett copula

def Gumbel_Barnett(XY, alpha):
    
    return XY.x+XY.y-1+(1-XY.x)*(1-XY.y)*np.exp(
        -alpha*np.log(1-XY.x)*np.log(1-XY.y)
        )

def inv_Gumbel_Barnett(u1, c, alpha):
    
    import scipy.optimize as so
    
    y = -np.log(1-u1)
    
    f = lambda x: 1-(1+alpha*x)*np.exp(-(1+alpha*y)*x)-c
    
    x = so.fsolve(f, [0]*len(u1))
    
    return 1-np.exp(-x), 1-np.exp(-y)

#%% tries
# =============================================================================
# 
# u1 = ss.uniform(0, 1).rvs(1000)
# 
# c = ss.uniform(0, 1).rvs(1000)
# 
# alpha = 0.2
# 
# u, v = inv_Gumbel_Barnett(u1, c, alpha)
# 
# plt.scatter(u, v, alpha = 0.8, s = 8)
# =============================================================================

#%% T tries

# =============================================================================
# from tools import Qn
# from tools import Cn
# from tools import Dn
# from tools import Q
# from Copulas.fgm import FGM
# 
# q = Q(u, v, FGM, [alpha])
# 
# qnC = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Cn
#         )
# 
# qnD = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Dn
#         )
# 
# print((abs(q)>1).sum(), (abs(qnC)>1).sum(),
#       (abs(qnD)>1).sum())
# 
# TC = ((q-qnC)**2).sum()
# TD = ((q-qnD)**2).sum()
# 
# print("\n" ,TC, "\n", TD)
# 
# #%%
# 
# q = Q(u, v, Gumbel_Barnett, [alpha])
# 
# qnC = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Cn
#         )
# 
# qnD = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Dn
#         )
# 
# TC = ((q-qnC)**2).sum()
# TD = ((q-qnD)**2).sum()
# 
# print((abs(q)>1).sum(), (abs(qnC)>1).sum(),
#       (abs(qnD)>1).sum())
# 
# print("\n" ,TC, "\n", TD)
# 
# =============================================================================
