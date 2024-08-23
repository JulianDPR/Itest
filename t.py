# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 07:19:19 2024

@author: Julian
"""

import pandas as pd

path = "C:/Users/julia/OneDrive/Desktop/TG/Imag/TG.xlsx"

conf = pd.read_excel(path, sheet_name = "CONF")

pot = pd.read_excel(path, sheet_name = "POT")


#%%

Conf = pd.melt(conf, id_vars = ["H_0","Dependencia"],
               value_vars =  [100, 1000, 30, 50, 500])

Conf.columns = ["H_0","Dependencia", "n", "Confianza"]

Conf.n = Conf.n.astype("int")

Conf = Conf.sort_values(["H_0", "Dependencia","n"])

Conf.n = Conf.n.astype("string")

Conf["id"] = Conf.H_0.str.cat([Conf.Dependencia, Conf.n], sep="-")

pot.n = pot.n.astype("string")

pot["id"] = pot.H_0.str.cat([pot.Dependencia, pot.n], sep="-")

#%%

Conf.id = Conf.id.str.replace("weak","1.Débil")
Conf.id = Conf.id.str.replace("moderate","2.Moderado")
Conf.id = Conf.id.str.replace("strong","3.Fuerte")

#%%

merge = pot.merge(
    Conf[["id","Confianza"]]
    , how = "left", on = "id")

merge = merge[['H_0', 'Dependencia',
               'n', 'clayton', 'fgm',
               'frank', 'gumbel_barnett',
       'gumbel_hougaard', 'Confianza']]

merge.columns = pd.MultiIndex.from_tuples(
    [('H_0',""), ('Dependencia',""),
                   ('n',""), ("Potencia",'clayton'), 
                   ("Potencia",'fgm'),
                   ("Potencia",'frank'),
                   ("Potencia",'gumbel_barnett'),
           ("Potencia",'gumbel_hougaard'),
           ("Confianza",'Confianza')]
    
    )


print(merge.loc[(merge["H_0",""]=="frank")&(
    merge["Dependencia",""]=="1.Débil"),
    [(        'H_0',                ''),
                ('Dependencia',                ''),
                (          'n',                ''),
                (   'Potencia',         'clayton'),
                (   'Potencia',             'fgm'),
                (   'Potencia',  'gumbel_barnett'),
                (   'Potencia', 'gumbel_hougaard'),
                (  'Confianza',       'Confianza')]].to_latex(
        multirow=True,
        multicolumn = True))



