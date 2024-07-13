import numpy as np
import pandas as pd
import scipy.stats as ss
import time as t
from tools import ranks
import matplotlib.pyplot as plt
from tools import u_v
from tools import Cn

from tools import Bootst
from tools import fitdist
from tools import max_ll
from tools import log_likelihood
import scipy.optimize as so
import scipy.stats as ss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

XY = pd.DataFrame({
    'x': [0.97, 0.54, 0.73, 0.17, 0.32],
    'y': [0.21, 0.18, 0.55, 0.36, 0.82]
})

u, v = u_v(XY)

u, v = np.meshgrid(np.sort(u), np.sort(v))

u, v = u.ravel(), v.ravel()

sns.heatmap(Cn(XY, u, v).reshape(-1,5)[::-1],
            xticklabels = np.array(range(1,6))/5,
            yticklabels = np.array(range(1,6)[::-1])/5,
            cmap = "flare",
            vmin = 0, vmax = 1,
            annot = True)

#%%


path = "C:/Users/Bienvenido/Desktop/TG/Imag/gumbel_hougaard/muestra.xlsx"

df = pd.read_excel(path, sheet_name = "strong")

par = max_ll(df, ss.genextreme, [0.94, 0.23], log_likelihood)

df["ranks"] = df.sort_values(by=0).reset_index().sort_values(by="index").reset_index()["level_0"]+1

df["u"] = df["ranks"]/df.shape[0]

df["f"] = ss.genextreme(0, *par).ppf(df["u"])

df.columns = ["x"]+list(df.columns[1:])

fig, ax = plt.subplots(figsize=(10,10))

sns.regplot(df, x="x", y="f", line_kws = dict(color="darkred", alpha=1, linewidth=100),
            scatter_kws=dict(color="darkcyan", alpha=0.8),
            marker=".", ax=ax, ci = 95)

ax.grid()


#%%
from tools import sample
import pandas as pd
from tools import Cn
from tools import Qn
from tools import Q
from Copulas.clayton import Clayton

a = 50
n = 10000

u,v = sample(n, "clayton", a)

qn = Qn(pd.DataFrame(dict(x=u,y=v)), u,v, Cn)

q = Q(u,v,Clayton, [a])

pi = u*v

pi_ = (1-u)*(1-v)

plt.hist(np.sqrt(n)*np.sqrt(pi*pi_)*(qn-q))

plt.show()

plt.hist(np.sqrt(n)*(qn-q))

plt.show()

#%% Cython proof
import time as t
import cython
from cython.parallel import prange


def psum(n: cython.int):

    i: cython.int
    sum: cython.int = 0

    for i in prange(n, nogil=True, num_threads = 1 if n<10000 else 8):
        sum += i

    return sum
     # Executed sequentially
print(psum(10000))     # Executed in parallel

#%%
from TG.gumbel_barnett import inv_Gumbel_Barnett_cython as Cinv_Gumbel_Barnett
import numpy as np
from tools import sample
import time as t

alpha = 0.5

a = t.time()
uc, vc = sample(1000, 'gumbel_barnett', alpha, cython=True)
b = t.time()
print((b-a)/60)

a=t.time()
up, vp = sample(1000, "gumbel_barnett", alpha)
b=t.time()
print((b-a)/60)

import matplotlib.pyplot as plt

plt.scatter(uc,vc,s=5)
plt.show()
plt.scatter(up,vp,s=5)
plt.show()