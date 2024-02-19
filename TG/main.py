# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:56:47 2023

@author: Julian
"""

#%% Import oficials packages
#Statics
import scipy.stats as ss
import statsmodels.api as sa
import statsmodels.stats as sts
#Linear algebra and dataframes
import numpy as np
import numpy.linalg as nl
import pandas as pd
#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
#Fit
from distfit import distfit
#
import time as t
import threading as th

#%% Own modules
#from tools
from tools import Cn
from tools import Dn
from tools import Qn
from tools import Q
from tools import ranks
from tools import T_gen
from tools import sample
from tools import fitdist
from tools import Escritor
from tools import t_gen

#%% T Generator
np.random.seed(1914)

n = 1000

path = "C:/Users/Bienvenido/Desktop/TG/Imag"

#%% Parametros de estudio

parameters = pd.DataFrame(
    [
     [0.5, 2, 8],
     [1.86, 5.73, 18.19],
     [1.25, 2, 5],
     [.2, .5, .9],
     [.2, .5, .9]],
    index = [
        "clayton",
        "frank", 
        "gumbel_hougaard",
        "fgm", 
             "gumbel_barnett"
             ],
    columns = [
                "weak",
               "moderate",
               "strong"
               ]
    ).T

parameters = parameters.to_dict()
#%%

if __name__ == "__main__":
    
    start = t.time()

    for i in parameters:
        
        for j in parameters[i]:
            
            #k = th.active_count()
            
            m = 1100#//k
            
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
                
            t_gen(m, n, i, parameters[i][j], results)
            
            T = (np.array(results).ravel())[:1000]
            
            fig, ax, results = fitdist(T, i+j)
            
            fig.savefig(path+"/"+i+"/"+j+".png", dpi = 500)
            
            fig.show()
            
            if j == "weak":
                
                Escritor(path+"/"+i+"/"+"resultados.xlsx", results["summary"], Name=j, index = False, 
                         mode = "w")
                
                Escritor(path+"/"+i+"/"+"muestra.xlsx", pd.Series(T), Name=j, index = False, 
                         mode = "w")
            
            else:
                
                Escritor(path+"/"+i+"/"+"resultados.xlsx", results["summary"], Name=j, index = False)
                
                Escritor(path+"/"+i+"/"+"muestra.xlsx", pd.Series(T), Name=j, index = False)
                
    end = t.time()

    print("Total Horas de trabajo: %.4f"%((end-start)/60/60))     


#%%

#import os

#os.system("shutdown -h")