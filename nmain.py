# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:18:59 2024

@author: Julian
"""

#%% Import oficials packages
#Statics
import scipy.stats as ss
import scipy.optimize as so
import statsmodels.api as sa
import statsmodels.stats as sts
#Linear algebra and dataframes
import numpy as np
import numpy.linalg as nl
import pandas as pd
import polars
#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
#Fit
from distfit import distfit
#
import time
import threading as th

#%% Own modules
#from tools
# Writer
from tools import Escritor
# Sample
from tools import sample
from tools import t_gen
# Statistics
from tools import Cn
from tools import Dn
from tools import Qn
from tools import Q
from tools import ranks
from tools import T_gen
from tools import k_t
# fit distribution
from tools import fitdist
from tools import fit
# Estimation methods
from tools import moms
#from tools import mom2
from tools import Bootst
#from tools import Bootst_2
from tools import bias
#from tools import bias2
from tools import gum_mom
from tools import bias_plot
# Calculus tools 
from findiff import FinDiff
from tools import c3
# Ttest
from tools import Ttest_
#from tools import Pow_Conf
from tools import T
from tools import Pow_Conf_
from tools import fitdist2
from tools import bias2
from tools import Bootst_2
from tools import bias_plot2

from Copulas.fgm import FGM
#%% Objetivo 1
path = "C:/Users/julia/OneDrive/Desktop/TG/Imag"

Tg = False
color = "#1b1b53"
GR = False

n = [
    30,
     50,
      100,
    500,
     1000
     ]

m = 1100#//k

names = [
   "gumbel_barnett",
    "clayton",
     "frank", 
     "gumbel_hougaard",
     "fgm"
         ]

dependence =  [
            "weak",
           "moderate",
           "strong"
           ]

parameters = pd.DataFrame(
    [
    [.2, .5, .9],
    [.5, 2, 8],
    [1.86, 5.73, 18.19],
    [1.25, 2, 5],
    [.2, .5, .9]
     ],
    index = names,
    columns = dependence
    ).T

parameters = parameters.to_dict()
#%% Showing

for i in parameters.keys():
    
    fig, ax = plt.subplots(1,3,figsize=(7,3))
    
    ax = ax.ravel()
    
    for j,k in zip(parameters[i].keys(),ax):
        #print(parameters[i][j])
        u,v = sample(1000, i, parameters[i][j])
        k.scatter(x=u, y=v,
                        marker=".",
                        facecolors="none",
                        edgecolors=color,
                        #label=r"$\theta = $"+f"{parameters[i][j]}",
                        s=30
                        )
        
        #k.legend(loc="lower right",fontsize=15, facecolor="none")
        k.set_xlabel(r"$\theta = $"+f"{parameters[i][j]}")
    fig.tight_layout()
    
    fig.savefig(path+"/"+i+"/"+f"{i}"+".png", dpi = 200)
    
    fig.show()
#%%

n_, name_, pars_ = np.meshgrid(n, names, dependence)

n_, name_, pars_ = n_.ravel(), name_.ravel(), pars_.ravel()

#%%

if __name__ == "__main__":
    
    T_samples = {}
    
    T_sample = {}

#%%
    
    #start = time.time()
    
    for i,j,k in zip(n_, name_, pars_):
        
        c = 0
        
        if j not in T_sample.keys():
            
            T_sample.update({j:{}})
        
        #k = th.active_count()
        
        # Create a results list to store the results
        results = []

        # Create and start a thread for each data chunk
        #threads = []
        
        #for p in range(k):
            
          #  thread = th.Thread(target=t_gen, args=(m, n, i, parameters[i][j], results, Cn))
            
          #  threads.append(thread)
            
          #  thread.start()
            
       # for thread in threads:
            
            #thread.join()

        t_gen(m, i, j, parameters[j][k], results, threading=Tg, random_seed=1927+c, GR=GR)
        
        T = (np.array(results).ravel())[:1000]
        
        T_samples.update({f"{i}_{k}_{j}":T})
        
        if k not in T_sample[j].keys():
        
            T_sample[j].update({k:{}})
        
        T_sample[j][k].update({f"{i}":T})
        
        #print(T_sample[j][k])
        
        fig, ax, results = fitdist(T, f"{i}_{k}_{j}", color = color)
        
        plt.close(fig)        
       # fig.savefig(path+"/"+j+"/"+f"{i}_{k}"+".png", dpi = 300)
        
       # fig.show()
        
        if f"{i}_{k}" == f"{min(n)}_weak":
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False, 
                     mode = "w")
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False, 
                     mode = "w")
        
        else:
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False)
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False)
        
        c += 1

#%% Objetivo 1.1

    espaniol = {"weak":"Débil", "moderate":"Moderada", "strong":"Fuerte"}

    dependencia, nombres, muestra = np.meshgrid(dependence, names, n)

    dependencia, nombres, muestra = dependencia.ravel(), nombres.ravel(), muestra.ravel()

    for i,j,k in zip(dependencia, nombres, muestra[::-1]):
        
        lim = 0
    
        if k == np.max(muestra):
        
            fig, ax = plt.subplot_mosaic([["left","left","right1","right2"],
                                      ["left", "left", "right3", "right4"]]
                                     , figsize=(12, 6))
        
            fitdist2(T_sample[j][i][f"{k}"], f"n={k} (Dependencia: {espaniol[i]})" ,ax["left"], color=color)
        
            ax["left"].set_xlabel("Valores")
            ax["left"].set_ylabel("Frecuencias")
        
        elif k > np.min(muestra):
        
            fitdist2(T_sample[j][i][f"{k}"],f"n={k}" ,
                     ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]], color = color)
          
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].set_ylabel("")
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].set_xlabel("")
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].legend(prop={'size': 4.5})
            
        
        else:
        
            fitdist2(T_sample[j][i][f"{k}"],f"n={k}" ,
                 ax[list(ax.keys())[-1]], color = color)
          
            ax[list(ax.keys())[-1]].set_ylabel("")
            ax[list(ax.keys())[-1]].set_xlabel("")
            
            ax[list(ax.keys())[-1]].legend(prop={'size': 4.5})
            #Ultima modificacion
            fig.suptitle("Distribución empírica del indicador"+r" $I^{2}$:"+f" {j}")
            
            fig.tight_layout()
        
            fig.savefig(path+"/"+j+"/"+f"Resumen_ajuste_{j}{i}"+".png", dpi = 100)
        
            fig.show()

#%% Ejemplo metodología
    how = "pot"
    #how = "pot"
    #only_sample = True
    only_sample = False

    fig, ax = plt.subplot_mosaic([["top"]*4,
                              ["bottom1","bottom2","bottom3","bottom4"]],
                              figsize=(16, 8))
    sns.histplot(T_sample["fgm"]["weak"]["100"], ax = ax["top"], color = color)
    ax["top"].set_title("Distribución de las réplicas del indicador")
    ax["top"].set_xlabel("")
    ax["top"].set_ylabel("Frecuencia absoluta", fontsize=12)
    b = [np.random.choice(T_sample["fgm"]["weak"]["100"], size = 1000) for i in range(4)]
    for i,j in zip(b, list(ax.keys())[1:]):
        
        if not how=="conf":
        #Potencia
            u, v = sample(100, "clayton", 1)
        else:
        #Confianza
            u,v = sample(100,"fgm",0.2)
        
        tc = ((Qn(pd.DataFrame(dict(x=u,y=v)), u, v,Cn, GR=GR)-Q(u, v, FGM, [.2], GR=GR)
               )**2).sum()
        #Si se quita cond del hue, se puede analizar solo las muestras
        g = pd.DataFrame([i,
            i>=np.quantile(i, 0.95)], index = ["Values", "Cond"]).T
        if not only_sample:
            sns.histplot(x="Values", data = g, hue = "Cond" ,ax = ax[j],
                      palette = [color, "darkred"])
            ax[j].vlines(np.quantile(i, 0.95),*ax[j].get_ylim(),
                     label= "Limite superior", color = "red")
            ax[j].vlines(tc,*ax[j].get_ylim(),
                     label= "Valor calculado", color = "green")
        else:
            sns.histplot(x="Values", data = g ,ax = ax[j],
                      color = color)
            
        ax[j].set_title(f"Remuestreo-(k) = ({list(ax.keys()).index(j)})")
        ax[j].set_xlabel("")
        ax[j].set_ylabel("")
    for i in ax.keys():
        ax[i].grid()
        ax[i].legend()
        for j in ax[i].spines:
            ax[i].spines[j].set_visible(False)
    fig.tight_layout()
    
    if only_sample:
    
        fig.savefig(path+"/"+"Demo.png", dpi=100)
        
    elif how=="conf":
        
        fig.savefig(path+"/"+"Demo2.png", dpi=100)
    
    else:
        
        fig.savefig(path+"/"+"Demo3.png", dpi=100)
        
    fig.show()


        
        
#%% Objetivo 2 Corregido

    df = pd.DataFrame(T_samples)
    
    Bias = bias2(df,1000)
    
    colors_ = ["#1b1b53", "#832727", "#246724"]
    
    for i in np.unique(name_):
    
        fig, ax = plt.subplots(2,1,figsize=(10,10))
    
        ax[0].set_title(r"Sesgo ${\bar{I}}^{(k)}-E[I]$")
    
        ax[1].set_title(r"MSE $E[({\bar{I}}^{(k)} - E[I])^{2}]$")
        
    
        espaniol2 = pd.Series({"weak":"Débil", "moderate":"Moderada", "strong":"Fuerte"}).str.replace("Débil","1.Débil").str.replace("Moderada","2.Moderada").str.replace("Fuerte","3.Fuerte")
    
        Sesgo = pd.pivot_table(Bias.loc[name_==i], values = "bias",
                           index = n_[name_==i], columns = espaniol2[pars_[name_==i]].values)
        
        Sesgo.columns = Sesgo.columns.str.replace(".","").str.replace(r"\d+","", regex=True)

        Sesgo.plot(kind="line", marker="o", ax=ax[0], rot = 90, color = colors_)
        
        ax[0].set_xticks(n)

        ECM = pd.pivot_table(Bias.loc[name_==i], values = "mse",
                           index = n_[name_==i], columns = espaniol2[pars_[name_==i]].values)
    
        ECM.columns = ECM.columns.str.replace(".","").str.replace(r"\d+","", regex=True)
        
        ECM.plot(kind="line", marker="o", ax=ax[1], rot = 90, color = colors_)
        
        ax[1].set_xticks(n)
        
        X0 = ax[0].get_xlim()
        
        ax[0].hlines(0, *X0, linestyle="--", color = "black")
        
        ax[0].grid()
        
        ax[0].set_xlim(X0)
        
        ax[0].set_ylabel("Sesgo")
        
        X1 = ax[1].get_xlim()
        
        ax[1].hlines(0, *X1, linestyle="--", color = "black")
        
        ax[1].grid()
        
        ax[1].set_xlim(X1)
        
        ax[1].set_ylabel("Error cuadrático medio")
        
        ax[1].set_xlabel(r"Tamaño de muestra $n$")
        
        fig.tight_layout()
        
        fig.savefig(f"{path}/{i}/ECM{i}.png", dpi=100)
        
#%% Objetivo 2 antiguo
# Cuales son las distribuciones que mejor se ajustan
# El sesgo de cada uno de los parametros de la distribucion
# La eficiencia y consistencia del parametro

    
    
    #Bias = bias(df)
    # #%%
    #
    
    # Bias_table = Bias["mse"].T[["Mln","Mgum"]].groupby([pd.Series(name_).str.replace("_", " ").str.title().values,espaniol2[pars_].values,n_]).apply(lambda x: ((x==x.min(axis=1).values.reshape(-1,1))).mean()).reset_index()
    # Bias_table.columns = ["Función Cópula", "Dependencia", "n", "Log-Normal", "Gumbel"]
    # Bias_table = pd.melt(Bias_table, Bias_table.columns[:3], Bias_table.columns[3:])
    # Bias_table.columns = list(Bias_table.columns[:3])+["Distribución", "Mínimo error cuadrático"]
    # Bias_table = pd.pivot_table(Bias_table, values = "Mínimo error cuadrático", columns = ["Distribución", "Dependencia"][::-1], index = ["Función Cópula","n"], aggfunc="sum")
    # print(Bias_table.to_latex(multirow=True,
    #                                                                              caption="Análisis individual de distribuciones ajustadas con menor error cuadrático medio. 1: El error cuadrático es el mínimo, 0: El error cuadrático no es el mínimo",
    #                                                                              label="tab:calidad", float_format="{:.0f}".format).replace(
    #                                                                                  "fgm","FGM").replace("1.","").replace("2.","").replace("3.","").replace(
    #                                                                                      "cline","cmidrule"))
    # print(Bias["mse"].T[["Mln","Mgum"]].groupby([pd.Series(name_).str.replace("_"," ").str.capitalize().values,espaniol2[pars_].values,n_]).apply(lambda x: ((x==x.min(axis=1).values.reshape(-1,1))).mean()).mean())
    #%%
    k = df.apply(moms).T
    
    print(k.groupby([pd.Series(name_).str.replace("_", " ").str.title().values,espaniol2[pars_].values,n_]).mean().round(2).to_latex(multirow=True,
                                                            caption="Valor estimado de los parámetros de la distribución candidata",
                                                            label="tab:estimacion", float_format="{:.1f}".format).replace(
                                                                "fgm","FGM").replace("cline","cmidrule").replace(
                                                                    "fgm","FGM").replace("1.","").replace("2.","").replace("3.",""))
    
   # print(k.groupby([name_,pars_]).median().round(2))
    
   # k.groupby([name_,pars_]).median().plot(kind="line",marker=".", rot=90)
    
    plt.show()
    
    aj = df.apply(lambda x: fit(x, k.loc[x.name,:]))

#%%
    AD_table = aj.T.groupby([pd.Series(name_).str.replace("_", " ").str.title().values,espaniol2[pars_].values,n_]).apply(lambda x: np.sum(x>0.05)).reset_index()

    AD_table.columns = ["Función Cópula", "Dependencia", "n", "Log-Normal", "Gamma", "Gumbel"]
            
    AD_table = pd.melt(AD_table, AD_table.columns[:3], AD_table.columns[3:])

    AD_table.columns = list(AD_table.columns[:3])+["Distribución", "Valor-p"]

    AD_table = pd.pivot_table(AD_table, values = "Valor-p", index = ["Función Cópula", "n"], columns = ["Distribución","Dependencia"][::-1], aggfunc="sum")

    print(AD_table.to_latex(multirow=True,
                           caption="Prueba de Anderson-Darling. 1: La prueba no rechaza la distribución, 0: La prueba rechaza la distribución",
        label="tab:Anderson", float_format="{:.3f}".format).replace(
            "cline","cmidrule").replace(
                "clayton","Clayton").replace("gumbel_barnett",
                "Gumbel Barnett").replace(
                    "frank","Frank").replace("gumbel_hougaard",
                    "Gumbel Hougaard").replace(
                        "fgm","FGM").replace(
                "ad_ln", "Log-Normal",
            ).replace("ad_gamma", "Gamma").replace("ad_gum", "AD Gumbel"
            ).replace("moderate","Moderado"
                      ).replace("weak","Débil"
                                ).replace("strong","Fuerte").replace("1.","").replace("2.","").replace("3.",""))
                                                                                                                                           
    
#%%

    #df.apply(lambda x: fit(x, k.loc[x.name,["ln_m","ln_b"]], True))
    
    #print((Bias["mse"].T.values[:,[0,2]] >= Bias["mse"].T.values[:,1][:,None]).mean(axis=0).mean())
    
    pars__, name__ = np.meshgrid(np.unique(pars_),np.unique(name_))
    
    name__, pars__ = name__.ravel(), pars__.ravel()
    
    result_ = {}
    
    for i,j in zip(name__, pars__):
        
        result_.update({i+"-"+j:bias_plot2(pd.DataFrame((pd.DataFrame(T_sample)).loc[j,
                                    i])**(1/2), ""
                                        #  i.upper()+"-"+j.upper()
                                          , path+"/"+i+"/"+f"{i}_{j}"+".png")})
    
#%% Objetica 3

# Intervalos de confianza (Distribución y empíricos)

    IE = df.quantile([0.025,0.975]).T


#%% Confidence

    prue_conf = pd.DataFrame(columns = ["n", 
                                        "H_0",
                                    "Dependencia",
                                        "Valor-p"])
    list_conf = []
    
    cond = 2
   
    for i,j,k in zip(n_, name_, pars_):
        
        for t in np.array(names)[np.isin(names,[j])]:
            
            while True:
            
                try:
                    
                    sample_uv = pd.DataFrame(sample(i, j,
                                                parameters[j][k]),
                                         index = ["x","y"]).T
                
                    list_conf.append([i, j, k, 
                              Pow_Conf_(sample_uv,
                                       tkey=t,
                                       m_resamples=1000, GR=GR)
                              ])
                    
                    cond = 2
                    
                    break
                
                except:
                    
                    cond -= 1
                    
                    print(cond)
                    
                    if cond == 0:
                    
                        cond = 2
                        
                        break
                    
            
    prue_conf = pd.DataFrame(np.array(list_conf), columns = prue_conf.columns)
     
    tab_conf = pd.pivot_table(prue_conf,
                            values = "Valor-p",
                            index = ["H_0",
                                     "Dependencia","n"],
                             aggfunc="sum")
#%% Power 

    cond = 2

    prue_pot = pd.DataFrame(columns = ["n", 
                                        "H_0",
                                        "Dependencia",
                                        "H_1",
                                        "Valor-p"])
    
    list_pot = []
 
    copula = "clayton"   
 
    for i,j,k in zip(n_[name_==copula],
                     name_[name_==copula], pars_[name_==copula]):
        
        
        for t in np.array(names)[~np.isin(names,[j])]:
            
            while True:
                    
                try:
                    
                    parameters_ = parameters[j][k]
                    
                    sample_uv = pd.DataFrame(sample(i, j,
                                                parameters_),
                                         index = ["x","y"]).T
                    
                    list_pot.append([i, j, k, t,
                             Pow_Conf_(sample_uv,
                                      tkey = j,
                                      m_resamples=1000,
                                      key = t, GR=GR)
                             ])
                    
                    cond = 3
                    
                    break
                
                except:
                    
                    try:
                        
                        parameters_ = parameters[j][k]
                        
                        if copula == "clayton":
                            
                            kt_ = lambda x: x/(x+2)
                            
                            g = lambda x: 2*x/(1-x)
                            
                            paramters_ = g(-kt_(parameters_))
                        
                        elif (copula == "gumbel_hougaard") or (copula == "gumbel_barnett"):
                            
                            parameters_ = parameters[j][k]
                        
                        else:
                            
                            parameters_ = -parameters[j][k]
                            
                            
                            
                                 
                        sample_uv = pd.DataFrame(sample(i, j,
                                                    parameters_),
                                             index = ["x","y"]).T
                        
                        list_pot.append([i, j, k, t,
                                 Pow_Conf_(sample_uv,
                                          tkey = j,
                                          m_resamples=1000,
                                          key = t, GR=GR)
                                 ])
                        
                        cond = 2
                        
                        break
                        
                    except:
                        
                        cond -= 1
                        
                        if cond == 0:
                            
                            cond = 2
                            
                            break
                    
                    cond -= 1
                    
                    #print(cond)
                    
                    if cond == 0:
                        
                        cond = 2
                        
                        break
                    
    prue_pot = pd.DataFrame(np.array(list_pot), columns = prue_pot.columns)
    
    tab_pot = pd.pivot_table(prue_pot,
                             values = "Valor-p",
                             index = ["H_0", "Dependencia","n"],
                             columns = ["H_1"], aggfunc="sum")
         
#%% Finish

ruta_salida = "C:/Users/julia/OneDrive/Desktop/TG/Imag/CALIDAD_TEST.xlsx"

try:
    Escritor(ruta_salida, tab_conf.reset_index(),
         f"CONFIANZA_{copula}", mode="a", index=False)

except:
    Escritor(ruta_salida, tab_conf.reset_index(),
         "CONFIANZA", mode="w", index=False)

Escritor(ruta_salida, tab_pot.reset_index(),
         f"POTENCIA_{copula}", mode="a", index=False)

