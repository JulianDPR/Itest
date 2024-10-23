 # -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:58:24 2023

@author: Julian
"""

#%% Import packages

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.stats as sta
import statsmodels.formula.api as sfa
import statsmodels.api as sa
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from findiff import FinDiff
import threading as th
from tqdm import tqdm, auto
from distfit import distfit

#%% Copulas modules
from Copulas.clayton import Clayton
from Copulas.gumbel_hougaard import Gumbel_Hougaard
from Copulas.fgm import FGM
from Copulas.gumbel_barnett import Gumbel_Barnett
from Copulas.frank import Frank

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

# def ranks(XY):
    
#     """
#     XY : DataFrame with n samples u and v
    
#     Return : The ranks statistics
#     of u and v and lenght sample
#     """
#     try: 
         
#         n = XY.shape[0]
        
#         #R = XY.sort_values(by="x").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
#         R = XY.x.rank(method="average").values
        
#         #S = XY.sort_values(by="y").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
#         S = XY.y.rank(method="average").values
        
#         return n, R, S
    
#     except:
        
#         raise Exception("DF with columns names (x,y)")
        
        
def ranks(XY, polars=False):
    
    """
    XY: DataFrame with n samples u and v
    Return: The ranks statistics of u and v,
    furthermore length sample
    polars: Default is False (DataFrame is pandas)
    """
    
    try:
        
        n = XY.shape[0]
        
        if polars:
            
            r = XY.select(
                pl.all().rank(
                method="average"
                                        )
                )
            
            return n, r["x"].to_numpy(), r["y"].to_numpy()
            
        else:
            
            R = XY.x.rank().values
            S = XY.y.rank().values
            
            return n, R, S
    except:
        
        raise Exception("DF with columns names (x,y)")
        
        

#%% Escale the sample
        
def u_v(XY, polars=False):
    
    """
    XY : DataFrame with n samples x and y
    
    Return : u and v
    """
    
    n, R, S = ranks(XY, polars)
    
    return R/(n), S/(n)    
    

#%% Empirical copula

# Biased

def Dn(XY, u, v, polars=False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    return: Classic empirical cumulative distribution function
    
    """
    
    n, R, S = ranks(XY, polars)
    
    dn = np.sum((R / (n) <= u.reshape(-1,1)) & (S / (n) <= v.reshape(-1,1)),
                axis=1)/n
    
    return dn
    
# Unbiased

def Cn(XY, u, v, polars=False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    return: Modificatedt empirical cumulative distribution function
    
    """
    
    n, R, S = ranks(XY, polars)
    
    cn = np.sum((R / (n + 1) <= u[:, None]) & ((S / (n + 1)) <= v[:, None]),
                axis=1)/n
    
    return cn

#%% Empirical q

def Qn(XY, u, v, function, GR = False, polars=False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    functon: The empirical Copula that you want to use Cn or Dn
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    fn = function(XY, u, v, polars)
    
    if GR:
        
        return fn
    
    else:
        
        return (fn-pi)/w

#%% Real q

def Q(u, v, Cfunction, parameters, diff = False ,GR = False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    cfuncton: The real copula that you want to evalue
    
    parameters: A vectro with estimate parameters
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    f = Cfunction(pd.DataFrame([u,v],index=['x','y']).T, *parameters, diff)
    
    
    if GR:
        
        return f
    
    else:
        
        return (f-pi)/w


#%% Sample generator

def sample(n, key, alpha, threading = True,
           cython = False, n_threat = 8):
    
    """
    n: u, v sample size
    
    key: Copula generate name
    
    alpha: Copula real parameter
    
    return: u, v sample
    
    """
        
    v1 = ss.uniform(0,1).rvs(n)
    
    v2 = ss.uniform(0,1).rvs(n)
    
    if key == 'clayton':
        
        from Copulas.clayton import inv_Clayton
        
        alpha = alpha/(-1<=alpha<float("inf"))
        
        u = v1
        
        v = inv_Clayton(u, v2, alpha)
        
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import inv_Gumbel_Hougaard
        
        alpha = alpha/(alpha>=1)
        
        x = ss.levy_stable(1/alpha,
                            1,
                            alpha==1,# Si ocurre entonces 1, caso contrario 0
                    np.cos(np.pi/(2*alpha))**alpha).rvs(n)
        
        u, v = inv_Gumbel_Hougaard(v1, v2, x, alpha)
        
    elif key == "frank":
        
        from Copulas.frank import inv_Frank
        
        alpha = alpha/(((alpha>-float("inf") and alpha < float("inf"))))
        
        u = v1
        
        v = inv_Frank(u, v2, alpha)
        
    elif key == "gumbel_barnett":
        
        #if cython:
         #   from TG.gumbel_barnett import inv_Gumbel_Barnett_cython as inv_Gumbel_Barnett    
        
        from Copulas.gumbel_barnett import inv_Gumbel_Barnett
        
        alpha = alpha/((alpha>=0) and (alpha<=1))
        
        n__ = n
        
        while n__%n_threat:
            
            n__ += 1   
            
        #print(n__)
        
        v1 = ss.uniform(0,1).rvs(n__)
        
        v2 = ss.uniform(0,1).rvs(n__)
        
        Th = []
        
        result = {}

        k = n_threat
        
        n_ = (n__)//k
        
        if threading:
        
            def quick(g, x, i):
                
                x.update({i:inv_Gumbel_Barnett(v1[g[0]:g[1]],
                                        v2[g[0]:g[1]],
                                        alpha)})

            for i in range(k):
            
                thread = th.Thread(target = quick,
                               args = ([n_*i, n_*(i+1)],
                                       result, i))
                
                Th.append(thread)
        
                thread.start()
            
            for i in Th:
            
                i.join()
        
           # print(result)
           
            result = [result[i] for i in np.sort(list(result.keys()))]
            
            result = np.array(result).ravel()
        
            u = v1[:n]
        
            v = result[:n]
            
# =============================================================================
#         
#             u = u[abs(v)<=1][:n]
#             
#             v = v[abs(v)<=1][:n]
# =============================================================================
        
            #print(u, v)
        
        else:
            
            u = v1            
                   
            v = inv_Gumbel_Barnett(v1, v2, alpha)
            
            
            
            #print(u, v)
        
    elif key == "fgm":
        
        from Copulas.fgm import inv_FGM
        
        alpha = alpha/(-1<= alpha <= 1)
        
        u, v = inv_FGM(v1, v2, alpha)
        
    else:
        
        raise Exception("La copula no esta dentro del estudio")
        
    #print(len(u), len(v), n__)    
      
    return u,v
#%% max log-likehood
def Diff(u, v, g, h=1e-6):
    
    return (g(u+h,v)-g(u-h,v))/(2*h)

def Diff_(alpha, u, v, g,h=1e-6):
    
    return (g(u,v, alpha+h)-g(u,v, alpha-h))/(2*h)

def c_density(u, v, function, alpha, h = 1e-6, diff = False, axis = 2):
    
    f = lambda u, v: function(pd.DataFrame(dict(x=u, y=v)), alpha, diff)
    
    d_u = lambda v, u: Diff(u, v, f, h)
    
    if axis == 1:
        
        return d_u(v,u)
                
    else:
    
        return Diff(v, u, d_u, h)

def inv_copula(u, c, function, alpha, h = 1e-6):
    
    f = lambda u,v: function(pd.DataFrame(dict(x=u,
                                               y=v),
                        index = list(range(len(np.array(np.ravel([u])))))),
                             alpha)
    
    d_u = lambda v,u,c: (Diff(u,v,f,h)-c).values[0]
    
    V = []
    
    for i,j in zip(u,c):
        
        V.append(so.root_scalar(d_u, args=(i, j), bracket = (0, 1), method="brenth" ,xtol=1e-6).root)
    
    return np.array(V).ravel()

def c(u, v, function ,alpha):
    
    #import inspect
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: Density copula function
    
    """
    if len(np.array([u]).ravel())==1:
        
        F = lambda x, y, a: function(pd.DataFrame(dict(x=x,y=y), index = [0]), a)
    
    else:
        
        F = lambda x, y, a: function(pd.DataFrame(dict(x=x,y=y)), a)
    
    d_x = nd.Derivative(F, method = "backward")
    
    d_x_y = nd.Derivative(lambda y, x, a: d_x(x, y, a), method = "backward")
    
    f = d_x_y(v, u, alpha)
    
    #f[np.isnan(f)] = 0

    return f#[~np.isnan(f)]

def c2(u ,v, function, alpha):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: Density copula function
    
    """
    
    du = u[1]-u[0]
    
    #F = lambda x, y, a: function(pd.DataFrame(dict(x=x, y=y)), a, diff = True)
    
    d_xy = FinDiff((0, du), (1, du))
    
    return d_xy(function(pd.DataFrame(dict(x=u, y=v), index = list(range(len(u)))), alpha, diff=True))


def c3(u, v, function, alpha, d = 1e-6, diff = False):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    d: Differencial
    
    return: Density copula function
    
    """
    
    #d = 1e-6
    
    f = lambda x, y: function(pd.DataFrame(dict(x=x, y=y), index = list(range(len(np.array(np.ravel([u])))))),
                              alpha, diff) 
    
    d_xy = lambda x, y, d : (((f(x+d,y+d)-f(x-d,y+d))/(2*d))-((f(x+d,y-d)-f(x-d,y-d))/(2*d)))/(2*d)
    
    return d_xy(u,v,d)

#%%
def mom(u, v, function, alpha):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: The moment estimated
    
    """
    
    return c(u, v, function, alpha).mean()
    
def clog_likelihood(alpha, function, XY):
    
    """
    XY: random vector uniform variables from u, v
    
    function: Real copula function
    
    return: The log-likelihood function
    
    """
    
    return -np.sum(np.log(c(XY.x, XY.y, function, alpha)))

def log_likelihood(theta, function, sample):
    
    """
    
    sample: m samples from the T distribution
    
    function: possible density function for T
    
    theta: parameters vector
    
    """
    
    if function == ss.genextreme:
    
        return -np.sum(np.log(function(0, theta[0], theta[1]).pdf(sample)))
    
    elif function == ss.gamma:
        
        return -np.sum(np.log(function(theta[0], 0, theta[1]).pdf(sample)))
    
    elif function == ss.lognorm:
        
        return -np.sum(np.log(function(0, theta[0], theta[1]).pdf(sample)))

def max_ll(XY, function, alpha, fun = clog_likelihood):
    
    """
    XY: random vector uniform variables from u, v
    
    function: Real copula function
    
    alpha: Seed of posible copula real parameter
    
    fun: The log-likelihood function
    
    return: Return the max(alpha) for the select copula
    
    """
    
    return np.array(so.fmin(fun, alpha, args=(function, XY)))

#u, v = sample(1000, 'clayton', 3)

#max_ll(pd.DataFrame([u, v], index=['x','y']).T, Clayton, 2)
#%% Integral

def trapezium_rule(values):
    
    """
    values: bidimensional matrix with u,v date
    return: integral result
    """
    
    B = values.copy()
    
    B[:, 1:-1] = B[:, 1:-1]*2
    
    B[1:-1, :] = B[1:-1, :]*2
    
    g = np.nansum(B)
    
    
    #if na_omit:
     #   f = (values[:,0] + 2*np.nansum(values[:,1:-1], axis=1) + values[:,-1]).ravel()
    #
     #   g = f[0] + 2*np.nansum(f[1:-1]) + f[-1]
    
    #else:
     #   f = (values[:,0] + 2*np.sum(values[:,1:-1], axis=1) + values[:,-1]).ravel()
    
      #  g = f[0] + 2*np.sum(f[1:-1]) + f[-1]
    
    return (1/(2*values.shape[0])**2) * g
    

def CopulaIM(function, alpha, nint = 300):
    
    """
    function: whatever bidimensional function what you want integrate
    alpha: parameters of the function
    nint: Numbers of space divisions
    return: trapezium_rule
    """
    
    x,y = np.linspace(1e-5,
                       1,
                       nint, endpoint=False),np.linspace(1e-5,
                                         1, nint, endpoint=False)
        
    return trapezium_rule(function(pd.DataFrame(dict(x=x, y=y)),
                    alpha, diff = True)*c2(x, y,
                             function,
                             alpha))
                                           
def CopulaIM2(function, alpha, nint = 300):
    
    x,y = np.linspace(1e-5,
                       1,
                       nint, endpoint=False),np.linspace(1e-5,
                                         1, nint, endpoint=False)
                                         
    return trapezium_rule(function(pd.DataFrame(dict(x=x, y=y)),
                    alpha, diff = True)*c3(x, y,
                             function,
                             alpha,diff=True))                                 
    
#%% T generator


def T(XY, u, v, function ,alpha, GR = False):
    """
    XY: Sample
    u: Rank of X
    v: Rank of Y
    function: Copula function
    alpha: Parameter
    
    return: Statistic
    """
    
    qn = Qn(XY, u, v, Cn, GR=GR)
    
    q = Q(u, v, function, [alpha], GR=GR)
    
    if not GR:
    
        qn_ = qn[(abs(q)<=1)&(abs(qn)<=1)]
    
        q_ = q[(abs(q)<=1)&(abs(qn)<=1)]
   # fitdist(np.sqrt(len(q_))*(qn_-q_))
        t_c = ((q_-qn_)**2).sum()
    
    else:
        
        t_c = ((q-qn)**2).sum()
    
    return t_c

def T_gen(n, key, alpha, empcop = Cn,
          threading = True, dist_key = None,
          dist_parameter = None,
          graph = False, GR = False,
          n_threat=1, cython = False):
    
    """
    n: u, v sample size
    
    Key: Copula name
    
    alpha: Copula real parameter
    
    empcop: The empirical Copula that you want to use Cn or Dn
    
    return: A sample from T distribution
    
    """
    
    if dist_key == None:
        
        dist_key = key
        
        dist_parameter = alpha
    
    u, v = sample(n, key, alpha, threading,
                  n_threat=n_threat, cython=False)
    
    XY = pd.DataFrame(dict(x = u, y = v))

    qn = Qn(XY, u, v, empcop)
    
    if dist_key == "clayton":
        
        from Copulas.clayton import Clayton
        
        function = Clayton
    
    elif dist_key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import Gumbel_Hougaard
    
        function = Gumbel_Hougaard
    
    elif dist_key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import Gumbel_Barnett
        
        function = Gumbel_Barnett
        
    elif dist_key == "fgm":
        
        from Copulas.fgm import FGM
        
        function = FGM
        
    elif dist_key == "frank":
        
        from Copulas.frank import Frank
        
        function = Frank
        
    else:
        
        raise Exception("La función no está dentro del estudio")

    # q = Q(u, v, function, [dist_parameter])
    
    # qn_ = qn[(abs(q)<=1)&(abs(qn)<=1)]
    
    # q_ = q[(abs(q)<=1)&(abs(qn)<=1)]
    
    # u_ = u[(abs(q)<=1)&(abs(qn)<=1)]
    
    # v_ = v[(abs(q)<=1)&(abs(qn)<=1)]
    
    # pi = u_*v_
    
    # pi_ = (1-u_)*(1-v_)
    
    # if graph:
        
    #     fitdist((q_-qn_)**2, "T sample")
    
    t = T(XY, u, v, function, alpha, GR)
    
    if graph:
        
        fitdist(T, "T sample")
    
    return t
    
   # fitdist(np.sqrt(len(q_))*(qn_-q_))
   
    # if GR:
    
    #     return (pi*pi_*(q_-qn_)**2).sum()
       
    # else:
        
    #     return ((q_-qn_)**2).sum()


def t_gen(m, n, key, alpha, return_ ,empcop = Cn,
          threading = False,
          GR = False, random_seed = 1927, tqdm_ = None,
          n_threat=1, cython = False):
    
    """
    m: T sample size
    
    n: u, v sample size
    
    Key: Copula name
    
    alpha: Copula real parameter
    
    return_: Save array
    
    empcop: The empirical Copula that you want to use Cn or Dn
    
    return: m samples from T distribution
    
    """
    
    np.random.seed(random_seed)
    
    if tqdm_ :
        
        return_.append(list(map(lambda x, y: T_gen(n, key,
                                                   alpha,
                                                Cn, threading,
                                                GR = GR,
                            n_threat=n_threat, cython=cython), 
                                range(m),
                                tqdm(range(m-1)))))
    else:
        
        return_.append(list(map(lambda x: T_gen(n, key, alpha,
                                            Cn, threading,
                                            GR = GR,
                                            n_threat=n_threat,
                                            cython=cython),
                                range(m))))

#%% Distfit

def fitdist(sample, title = None, color = "#647c8c"):
    
    """
    sample: m samples form T distristribution
    
    Title: Name for plot or file with copula names
    
    return: A set with posibles distributions for T
    
    """
    
    fig, ax = plt.subplots(figsize=(20,15))
    
    dfit = distfit()
    
    results = dfit.fit_transform(sample)
    
    dfit.plot(n_top = 3, ax = ax,
              bar_properties={'color': f'{color}'})
    
    if title != None:
        
        ax.set_title(title, fontsize = 16)
    
    return fig, ax, results

def fitdist2(sample, title="", axis = None, color = "#647c8c"):
    
    """
    sample: m samples form T distristribution
    
    Title: Name for plot or file with copula names
    
    return: A set with posibles distributions for T
    
    """
    
    if axis == None:
        
        return distfit(sample, title)
    
    else:
        
        dfit = distfit()
        
        results = dfit.fit_transform(sample)
        
        dfit.plot(n_top = 3, ax = axis,
                  bar_properties={'color': f'{color}'})
        
        axis.set_title(title, fontsize = 16)
        
        return results


    
#%% Bootstrapping 
def Bootst(sample, k):
    
    np.random.seed(1914)
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A df with bias, var and mse
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    T = pd.DataFrame(bootst).apply(moms)
    
    ET = T.mean(axis = 1)
    
    VT = T.var(axis = 1)
    
    
    VT = pd.Series(dict(Vgamma = VT.iloc[0]+VT.iloc[1],
              Vln = VT.iloc[2]+VT.iloc[3],
              Vgum = VT.iloc[4]+VT.iloc[5]))

    b = pd.concat([ET,VT])
    
    return b

def Bootst_2(sample, k):
    
    np.random.seed(1914)
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A df with bias, var and mse
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    T = pd.DataFrame(bootst).mean(axis=0)
    
    #print(T)
    
    b = pd.Series(dict(E = T.mean() ,
             V = T.var()))
    
    
   # VT = pd.Series(dict(Vgamma = VT.iloc[:3].sum(),
    #          Vln = VT.iloc[3:6].sum(),
     #         Vgum = VT.iloc[6:].sum()))

   # b = pd.concat([ET,VT])
    
    return b

def Bootst2(sample, k):
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A array of k size from the T distribution average resamples
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    b = np.quantile(bootst, np.array([0.025, 0.975]), axis = 0)
    
    return stat(b, axis = 1)

def Bootst3(sample, k):
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A array of k size from the T distribution average resamples
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    b = np.mean(bootst, axis = 0)
    
    return b

def Bootsttrap(sample, k):
    
    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    return bootst
    
#%% Estimation

def moms(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: Gamma, Gumbel, Lognormal parameters estimators
    
    """
    
    dic = dict(
    
    gamma_a = np.mean(sample)**2/(np.var(sample, ddof=1)),
    
    gamma_b = np.mean(sample)/(np.var(sample, ddof=1)),
    
    ln_m = -(1/2)*np.log((np.var(sample,
                                 ddof=1)+
                          np.mean(sample)**2)/(np.mean(sample)**4)),
    ln_b = np.log((np.var(sample,
                          ddof=1)+
                   np.mean(sample)**2)/np.mean(sample)**2),
    
    gum_m = np.mean(sample)-np.sqrt(6)*np.euler_gamma*np.std(sample,
                                                ddof=1)/np.pi,
    gum_b = np.std(sample, ddof = 1)*np.sqrt(6)/np.pi
    
    )
    
    #print(np.mean(sample))
    
    return pd.Series(dic)

# =============================================================================
# def mom2(sample):
#     
#     """
#     sample: A array with n repeats from x's distribution
#     
#     return: Gamma, Gumbel, Lognormal parameters estimators
#     
#     """
#     
#     gam = ss.gamma.fit(sample)
#     
#     gen = ss.genextreme.fit(sample)
#     
#     ln = ss.lognorm.fit(sample)
#     
#     dic = dic = dict(
#     
#     gamma_a = gam[0],
#     
#     gamma_b = gam[1],
#     
#     gamma_c = gam[2],
#     
#     ln_m = ln[0],
#     ln_b = ln[1],
#     ln_c = ln[2]
#     ,
#     gen_a = gen[0],
#     gen_b = gen[1],
#     gen_c = gen[2]
#     
#     )
#     
#     return pd.Series(dic)
# =============================================================================
    

def gum_mom(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: The Gumbel parameters estimators
    
    """
    
    dic = dict(
        gum_m = np.mean(sample)-np.sqrt(6)*np.euler_gamma*np.std(sample,
                                                    ddof=1)/np.pi,
        gum_b = np.std(sample, ddof = 1)*np.sqrt(6)/np.pi
        
        )
    
    return pd.Series(dic)

def bias(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: The Variance, Bias and Mse from X estimators
    """
    
    moms_ = sample.apply(moms)
    
    #print(moms_)
    
    bootst = sample.apply(lambda x: Bootst(x, 1000))
    
    r_bootst = bootst.loc[moms_.index,:]
    
    bias = (moms_-r_bootst)
    
    #print(bias)
    
    bias = pd.DataFrame(dict(
        Bgamma = bias.iloc[[0,1],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
        Bln = bias.iloc[[2,3],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
        Bgum = bias.iloc[[4,5],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2)
        )).T

    var = bootst.loc[bootst.index[6:],:]
    
    var_ = var.copy()
    
    var_.index = bias.index
    
    mse = bias+var_
    
    mse.index = ["Mgamma", "Mln", "Mgum"]
    
    return dict(bias = bias, var = var, mse = mse)

def bias2(sample, k):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: The Variance, Bias and Mse from X estimators
    """
    
    moms_ = sample.mean(axis=0)
    
    #print(moms_)
    
    bootst = sample.apply(lambda x: Bootst_2(x, k)).T
    
    #print(bootst)
    
    r_bootst = bootst.loc[moms_.index,:]
    
    bias = (moms_-r_bootst["E"])
    
    #print(bias)

    var = r_bootst["V"]
    
    var.index = bias.index
    
    mse = bias**2+var
    
    #mse.index = 
    
    return pd.DataFrame(dict(Media = moms_ , bias = bias, var = var, mse = mse))
    

def bias_plot(sample, title = "", path = None):
    """
    sample: A array with n repeats from x's distribution
    
    return: The plot of Variance, Bias and Mse from X estimators
    """
    
    Bias = bias(sample)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
    ax = ax.ravel()
    
    bias_ = (Bias["bias"].T[["Bln","Bgum"]])**(1/2)
    
    mse = Bias["mse"].T[["Mln","Mgum"]]
    
    n = np.array(list(bias_.index), dtype="float64").ravel()
    
    #print(n)
    
    (bias_).plot(kind = "line", marker = ".",
                ax = ax[0], color=["#1b1b53", "#832727", "#283d3d"])
    
    ax[0].set_title(r"Sesgo $\hat\lambda - E[\hat\Lambda]$")
    
    X0 = ax[0].get_xlim()
    
    ax[0].hlines(0, *X0, linestyle="--", color = "black")
    
    ax[0].grid()
    
    mse.plot(kind = "line", marker = ".", ax = ax[1],  color=["#1b1b53", "#832727", "#283d3d"])
    
    ax[1].set_title(r"MSE $E[({\hat\lambda}-\hat\Lambda)^{2}]$")
    
    X1 = ax[1].get_xlim()
    
    ax[1].hlines(0, *X1, linestyle="--", color = "black")
    
    ax[1].grid()
    
    fig.suptitle(title)
    
    fig.tight_layout()
    
    if path != None:
        
        fig.savefig(path, dpi = 200)
    
    fig.show()
    
    return Bias


def bias_plot2(sample, title = "", path = None, k = 1000):
    """
    sample: A array with n repeats from x's distribution
    
    return: The plot of Variance, Bias and Mse from X estimators
    """
    
    Bias = bias2(sample,k)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
    ax = ax.ravel()
    
    bias_ = Bias["bias"]
    
    mse = Bias["mse"]
    
    n = np.array(list(bias_.index), dtype="float64").ravel()
    
    #print(n)
    
    (bias_).plot(kind = "line", marker = ".",
                ax = ax[0], color=["#1b1b53", "#832727", "#283d3d"])
    
    ax[0].set_title(r"Sesgo $\hat{\bar{I}}^{(k)}-E[I]$")
    
    X0 = ax[0].get_xlim()
    
    ax[0].hlines(0, *X0, linestyle="--", color = "black")
    
    ax[0].grid()
    
    mse.plot(kind = "line", marker = ".", ax = ax[1],  color=["#1b1b53", "#832727", "#283d3d"])
    
    ax[1].set_title(r"MSE $E[(\hat{\bar{I}}^{(k)} - E[I])^{2}]$")
    
    X1 = ax[1].get_xlim()
    
    ax[1].hlines(0, *X1, linestyle="--", color = "black")
    
    ax[1].grid()
    
    fig.suptitle(title)
    
    fig.tight_layout()
    
    if path != None:
        
        fig.savefig(path, dpi = 200)
    
    fig.show()
    
    return Bias
    
    
# =============================================================================
# def bias2(sample):
#     
#     """
#     sample: A array with n repeats from x's distribution
#     
#     return: The Variance, Bias and Mse from X estimators
#     """
#     
#     moms_ = sample.apply(mom2)
#     
#     bootst = sample.apply(lambda x: Bootst_2(x, 1000))
#     
#     r_bootst = bootst.loc[moms_.index,:]
#     
#     bias = (moms_-r_bootst)
#     
#     bias = pd.DataFrame(dict(
#         Bgamma = bias.iloc[:3,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
#         Bln = bias.iloc[3:6,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
#         Bgum = bias.iloc[6:9,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2)
#         )).T
# 
#     var = bootst.loc[bootst.index[9:],:]
#     
#     var_ = var.copy()
#     
#     var_.index = bias.index
#     
#     mse = bias+var_
#     
#     mse.index = ["Mgamma", "Mln", "Mgum"]
#     
#     return dict(bias = bias, var = var, mse = mse)
# =============================================================================

def fit(sample, parameters, graph = False):
    
    if graph:
    
        fig, ax = plt.subplots(figsize=(6,6))
    
        qq = ss.probplot(sample, sparams=(parameters[1]**(1/2), 0, np.exp(parameters[0])),
                dist = ss.lognorm, plot = ax)
    
    #    np.random.seed(1914)
    
        ad = ss.anderson_ksamp([qq[0][1], qq[0][0]
                            ]).pvalue
    
        ax.set_title("AD-Pvalue: %.4f"%(ad))
    
        fig.suptitle(f"{sample.name}")
    
        ax.set_xlabel("Cuantil teórico")
    
        ax.set_ylabel("Cuantil ordenado")
    
        fig.tight_layout()
    
        fig.show()
    
    elif parameters.shape[0]>2:
        
        qq_ln = ss.probplot(sample, sparams=(parameters["ln_b"]**(1/2), 0, np.exp(parameters["ln_m"])),
                dist = ss.lognorm)
        
        qq_gam = ss.probplot(sample, sparams=(parameters["gamma_a"], 0, parameters["gamma_b"]),
                dist = ss.gamma)
        
        qq_gum = ss.probplot(sample, sparams=(0, parameters["gum_m"], parameters["gum_b"]),
                dist = ss.genextreme)
        
        ad = dict(ad_ln = ss.anderson_ksamp([qq_ln[0][1], qq_ln[0][0]
                            ]).pvalue,
                  ad_gamma = ss.anderson_ksamp([qq_gam[0][1], qq_gam[0][0]
                                                ]).pvalue, 
                  ad_gum = ss.anderson_ksamp([qq_gum[0][1], qq_gum[0][0]
                                                ]).pvalue)
        
        ad = pd.Series(ad)
        
    else:
        
        raise Exception("Parameters not be comparable")
    
    return ad


# =============================================================================
# def root(function, args):
#     
#     """
#     function: La función a la cual se le buscan las raíces
#     args: Argumentos extras de la función
#     regresa: El x donde la función es 0
#     """
#     
#     
# =============================================================================

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
#%% argmax

def kendall_tau(sample, parameter, cfunction):
    """
    parameter: Possible parameter from random sample
    sample: U and V array of values from any distribution
    cfunction: A possible copula function
    return: A Fastest estimation of kendall tau
    """
    
    return 4*np.mean(cfunction(sample, parameter))-1

def k_t(cfunction, parameter, nint = 1000):
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 4*CopulaIM(cfunction, parameter, nint)-1

def r_s(cfunction, parameter, nint=1000):
    
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 12*CopulaIM(cfunction, parameter, nint)-3


def k_t2(cfunction, parameter, nint = 1000):
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 4*CopulaIM2(cfunction, parameter, nint)-1

def r_s2(cfunction, parameter, nint=1000):
    
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 12*CopulaIM2(cfunction, parameter, nint)-3

def argmax(parameter, n, cfunction, kendalltau):
    
    """
    parameter: Possible parameter from random sample
    sample: U and V array of values from any distribution
    cfunction: A possible copula function
    kendaltau: Concordance non parameter estimation
    """   
    return k_t(cfunction, parameter)-kendalltau

def kv(key, kt):
    """
    key: Copula name
    kt: empirical kendall tau
    
    return: function and kendall bound, and
    parameter bound
    """
    
    if key == "clayton":
        
        from Copulas.clayton import Clayton
        
        function = Clayton
        
        lim = [-1, 1]
        
        plim = [-1, 50]
        
        #0, 1
    
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import Gumbel_Hougaard
    
        function = Gumbel_Hougaard
        
        lim = [0, 1]
        
        plim = [1, 50]
        
        #0, 1
    
    elif key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import Gumbel_Barnett

        function = Gumbel_Barnett
    
        lim = [0, -.361]
        
        plim = [0,1]
        
        #0, -.3612
        
    elif key == "fgm":
        
        from Copulas.fgm import FGM
        
        function = FGM
        
        lim = [-.222, .222]
        
        plim = [-1,1]
        
        # -0.222, 0.222
        
    elif key == "frank":
        
        from Copulas.frank import Frank
        
        function = Frank
        
        lim = [-1, 1]
        
        plim = [-36, 36]
        
        # -1, 1
        
    else:
        
        raise Exception("The function is out of study")
    
    return (function, lim, plim)

def maxkt(XY, n, function, kt, plim,
          lim, argmax = argmax):
    """
    n: sample size
    function: Copula function
    kt: Empirical kendall tau
    plim: parameter bound
    lim: kendall tau bound
    argmax: kendall maximun function
    
    return: alpha and a0
    
    """
   # print(function)
  
    if (min(lim) < kt < max(lim)):
        
        a0 = so.root_scalar(argmax,
                            args=(n, function, kt),
                            bracket=(plim[lim.index(min(lim))],
                                plim[lim.index(max(lim))])).root
        
        alpha = max_ll(XY, function, a0)[0]
        
    elif abs(min(lim)-kt)<=0.05:
        
        alpha = plim[lim.index(min(lim))]
        
        a0 = np.nan
        
    elif abs(max(lim)-kt)<=0.05:
        
        alpha = plim[lim.index(max(lim))]
        
        a0 = np.nan
    
    elif not (min(lim) < kt < max(lim)):
        
        raise Exception("Kendall tau out of limit")
    
    
    return (a0, alpha)
    
    
#%% T test

def Testimation(sample_, key, kt_ = None):
    
    """
    sample: DataFrame bivariate random sample
    key: Possible copula to estimate parameter
    return: Parameters estimation
    """
    if not kt_:
        
        kt = ss.kendalltau(*sample_)[0]
    
    else:
        
        kt = kt_
        
    #print(kt)
    
    function, lim, plim = kv(key, kt)
    
    XY = pd.DataFrame(sample_, index = ["x","y"]).T
    
    a0, alpha = maxkt(XY, XY.shape[0], function,
                      kt, 
                      plim,
                      lim)
    
    return XY, a0, alpha, function

def Ttest_(sample_, key, n_bootst = 1000,
           threading=False, GR = False):
    
    """
    sample: DataFrame bivariate random sample
    key: Possible copula
    n_bootst: Bootstrap sample from real values of statistic
    
    return: Sample statistic, Pvalue of statistic,
    Kendall's Tau, Copula Parameter
    
    """
    
    sample_ = np.array(u_v(sample_))
    
    XY, a0, alpha, function = Testimation(sample_, key)
    
    return_ = []
    
    t_gen(int(n_bootst*1.1), XY.shape[0],
          key, alpha, return_, threading=threading, GR=GR)
    
    t_sam = np.array(return_).ravel()[:n_bootst]
    
    t_c = T(XY, sample_[0], sample_[1], function,
            alpha, GR=GR)
    
    Ptc = (t_sam >= t_c).mean()
    
    return pd.DataFrame(dict(theta = [alpha], theta_t = [a0],
                             t_c = [t_c], 
                             Pvalue = [Ptc]))

def Pow_Conf_(sample_, tkey, confidence = 0.95,
              n_bootst = 1000, key = None,
              m_resamples = 1000, threading=False,
              GR=False):
    
    """
    sample: DataFrame bivariate random sample
    key: Possible copula
    n_bootst: Bootstrap sample from real values of statistic
    
    return: Sample statistic, Pvalue of statistic,
    Kendall's Tau, Copula Parameter
    """
    
    Sample = np.array(u_v(sample_))
    
    # tkey estimation 
    
    XY, a0, alpha, function = Testimation(Sample, tkey)
    
    #print(a0, alpha)
    
    return_ = []
    
    t_gen(int(m_resamples*1.1), XY.shape[0],
          tkey, alpha, return_, threading=threading,
          GR=GR)
    
    t_sam = np.array(return_).ravel()[:n_bootst]
    #-----------------
    t_bootst = Bootsttrap(t_sam, k=n_bootst)
    
    if not key:
        
        uv = list(map(lambda x: pd.DataFrame(sample(XY.shape[0], tkey, 
                                                     alpha),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x.values, x.y.values, function, alpha,
                                   GR=GR),
                       uv))
        
    
        #nbTest = (t_bootst>=np.array(t_c)).mean(axis=0)
    
        nbTest = Bootsttrap(np.quantile(t_bootst, confidence, axis=0), n_bootst)>Bootsttrap(np.array(t_c), n_bootst)
    
        #return (nbTest>(1-confidence)).mean()
        
        return nbTest.mean()
    
    else:
        
        #key estimation
        
        XY1, a1, alpha1, function1 = Testimation(Sample, key,
                                              k_t(function, alpha))
        
        
        uv = list(map(lambda x: pd.DataFrame(sample(XY1.shape[0], key, 
                                                     alpha1),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x.values, x.y.values,
                                   function, alpha, GR=GR),
                       uv))    
        #nbTest = (t_bootst>=np.array(t_c)).mean(axis=0)
    
        nbTest = Bootsttrap(np.quantile(t_bootst, confidence, axis=0), n_bootst) <= Bootsttrap(np.array(t_c), n_bootst)
    
        #return (nbTest<=(1-confidence)).mean()
        
        return nbTest.mean()
        
#%%
def conf_pow(Sample,t_sam, n_bootst,
             alpha, function, tkey,
             key = None,
             confidence = 0.95,
             threading=False):
    
    t_bootst = Bootsttrap(t_sam, k=n_bootst)
    
    n = Sample.shape[0]
    
    if not key:
        
        uv = list(map(lambda x: pd.DataFrame(sample(n, tkey, 
                                                     alpha),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x.values, x.y.values, function, alpha),
                       uv))
        
        #nbTest = (t_bootst>=np.array(t_c)).mean(axis=0)
    
        nbTest = np.quantile(t_bootst, confidence, axis=0)>np.array(t_c)
    
        #return (nbTest>(1-confidence)).mean()
        
        return nbTest.mean()
    
    else:
        
        #key estimation
        
        XY1, a1, alpha1, function1 = Testimation(Sample, key,
                                              k_t(function, alpha))
        
        
        uv = list(map(lambda x: pd.DataFrame(sample(n, key, 
                                                     alpha1),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x.values, x.y.values, function, alpha),
                       uv))    
        #nbTest = (t_bootst>=np.array(t_c)).mean(axis=0)
    
        nbTest = np.quantile(t_bootst, confidence, axis = 0) <= np.array(t_c)
    
        #return (nbTest<=(1-confidence)).mean()
        
        return nbTest.mean()