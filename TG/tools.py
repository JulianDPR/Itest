# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:58:24 2023

@author: Julian
"""

#%% Import packages

import numpy as np
import pandas as pd
import statsmodels.stats as sta
import statsmodels.formula.api as sfa
import statsmodels.api as sa
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

#%% Copulas modules
from Copulas.clayton import Clayton
from Copulas.gumbel_hougaard import Gumbel_Hougaard
from Copulas.fgm import FGM
from Copulas.gumbel_barnett import Gumbel_Barnett
from Copulas.frank import Frank
#%%
from distfit import distfit

#%% Otras

def Escritor(Doc,DataFrame,Name,index=True,over_repla="overlay",srow=0,scol=0,mode="a"):
  #Doc: Ruta
  #DataFrame: Datos
  #Name: Nombre del sheet
  #index: Incluir index ó no, por defecto es True
  #over_repla: Sobreescribir ó reemplazar
  if mode == "a":
    writer_ = pd.ExcelWriter(Doc, mode=mode,if_sheet_exists=over_repla)
  else:
    writer_ = pd.ExcelWriter(Doc, mode=mode)
  DataFrame.to_excel(writer_, sheet_name=Name,index=index, startrow = srow, startcol = scol)
  writer_.close()

#%% Ranks

def ranks(XY):
    
    """
    XY : DataFrame with n samples u and v
    
    Return : The ranks statistics
    of u and v and lenght sample
    """
    try: 
         
        n = XY.shape[0]
        
        #R = XY.sort_values(by="x").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
        R = XY.sort_values(by='x').reset_index().sort_values(by='index').reset_index().groupby('x')[['level_0']].transform('mean').values.ravel()+1
        
        #S = XY.sort_values(by="y").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
        S = XY.sort_values(by='y').reset_index().sort_values(by='index').reset_index().groupby('y')[['level_0']].transform('mean').values.ravel()+1
        
        return n, R, S
    
    except:
        
        raise Exception("DF with columns names (x,y)")

#%% Escale the sample
        
def u_v(XY):
    
    """
    XY : DataFrame with n samples x and y
    
    Return : u and v
    """
    
    n, R, S = ranks(XY)
    
    return R/n, S/n    
    

#%% Empirical copula

# Biased

def Dn(XY, u, v):
    
    n, R, S = ranks(XY)
    
    dn = ((R/n <= u)*(S/n <= v)).sum()/n
    
    return dn
    
# Unbiased

def Cn(XY, u, v):
    
    n, R, S = ranks(XY)
    
    cn = ((R/(n+1) <= u)*(S/(n+1) <= v)).sum()/(n)
    
    return cn

#%% Empirical q

def Qn(XY, u, v, function):
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    function = np.vectorize(function, excluded=[0])
    
    fn = function(XY, u, v)
    
    return (fn-pi)/w

#%% Real q

def Q(u, v, Cfunction, parameters):
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    f = Cfunction(pd.DataFrame([u,v],index=['x','y']).T, *parameters)
    
    return (f-pi)/w


#%% Sample generator

def sample(n, key, alpha):
    
    v1 = ss.uniform(0,1).rvs(n)
    
    v2 = ss.uniform(0,1).rvs(n)
    
    if key == 'clayton':
        
        from Copulas.clayton import inv_Clayton
        
        u = v1
        
        v = inv_Clayton(u, v2, alpha)
        
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import inv_Gumbel_Hougaard
        
        x = ss.levy_stable(1/alpha,
                            1,
                            alpha==1,# Si ocurre entonces 1, caso contrario 0
                    np.cos(np.pi/(2*alpha))**alpha).rvs(n)
        
        u, v = inv_Gumbel_Hougaard(v1, v2, x, alpha)
        
    elif key == "frank":
        
        from Copulas.frank import inv_Frank
        
        u = v1
        
        v = inv_Frank(u, v2, alpha)
        
    elif key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import inv_Gumbel_Barnett
        
        u, v = inv_Gumbel_Barnett(v1, v2, alpha)
        
    elif key == "fgm":
        
        from Copulas.fgm import inv_FGM
        
        u, v = inv_FGM(v1, v2, alpha)
        
    else:
        
        raise Exception("La copula no esta dentro del estudio")
        
    return u,v
#%% max log-likehood

def c(u, v, function ,alpha):
    
    f = np.vectorize(lambda u,v: nd.Hessian(lambda XY: function(pd.DataFrame([XY[0],XY[1]], index=['x','y']).T, alpha))([u,v]).diagonal(1))(u,v).ravel()

    return f[~np.isnan(f)]

def mom(u, v, function, alpha):
    
    return c(u, v, function, alpha).mean()
    
def log_likehood(alpha, function, XY):
    
    return -np.sum(np.log(c(XY.x, XY.y, Clayton, alpha)))

def max_ll(XY, function, alpha, fun = log_likehood):
    
    return so.fmin(fun, alpha, args=(function, XY))

#u, v = sample(1000, 'clayton', 3)

#max_ll(pd.DataFrame([u, v], index=['x','y']).T, Clayton, 2)


#%% T generator

def T_gen(n, key, alpha, empcop = Cn):
    
    u, v = sample(n, key, alpha)
    
    XY = pd.DataFrame([u, v], index = ["x", "y"]).T

    qn = Qn(XY, u, v, empcop)
    
    if key == "clayton":
        
        from Copulas.clayton import Clayton
        
        function = Clayton
    
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import Gumbel_Hougaard
    
        function = Gumbel_Hougaard
    
    elif key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import Gumbel_Barnett
        
        function = Gumbel_Barnett
        
    elif key == "fgm":
        
        from Copulas.fgm import FGM
        
        function = FGM
        
    elif key == "frank":
        
        from Copulas.frank import Frank
        
        function = Frank
        
    else:
        
        raise Exception("La función no está dentro del estudio")

    q = Q(u, v, function, [alpha])
    
    qn_ = qn[(abs(q)<=1)&(abs(qn)<=1)]
    
    q_ = q[(abs(q)<=1)&(abs(qn)<=1)]
    
    
    return ((q_-qn_)**2).sum()
    
#%% Distfit

def fitdist(sample, title = None):
    
    fig, ax = plt.subplots(figsize=(20,15))
    
    dfit = distfit()
    
    results = dfit.fit_transform(sample)
    
    dfit.plot(n_top = 3, ax = ax)
    
    if title != None:
        
        ax.set_title(title, fontsize = 16)
    
    return fig, ax, results

#%%

def t_gen(m, n, key, alpha, return_ ,empcop = Cn):
    
    return_.append(list(map(lambda x: T_gen(n, key, alpha, Cn), range(m))))

    
#%% Bootstrapping 


#%% Jack knife



# =============================================================================
# #%% Try
# 
# xy = {"x":ss.norm(0,1).rvs(100), "y":ss.norm(0,2).rvs(100)}
# 
# df = pd.DataFrame(xy)
# 
# print(Dn(df, 0.8, 0.8))
# print(Cn(df, 0.8, 0.8))
# print(Qn(df, 0.999, 0.999, Cn))
# 
# #%%
# 
# xs, ys = np.meshgrid(np.linspace(0.001,0.999,100), np.linspace(0.001,0.999,100))
# 
# qn = Qn(df, xs, ys, Dn)
# 
# DF = pd.DataFrame([xs.ravel(), ys.ravel(),
#                    qn.ravel()], index = ['X', 'Y', 'Qn']).T
# 
# DF.Qn = DF.Qn.apply(lambda x: np.nan if abs(x)>1 else x)
# 
# #%%
# 
# sns.heatmap(DF.Qn.to_numpy().reshape(*qn.shape), vmin=-1, vmax=1)
# 
# 
# =============================================================================


#%% basura

def cn_gen(n):
    
    u = ss.uniform(0,1).rvs(n)
    v = ss.uniform(0,1).rvs(n)
    
    return np.vectorize(Cn, excluded=[0])(pd.DataFrame(dict(x=u,y=v)), u, v)

