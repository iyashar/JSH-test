#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from scipy.stats import norm
from matplotlib import pyplot as plt


# In[13]:


def jsd(p, q, base=np.e):
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p+q)
    return sp.stats.entropy(p,m, base=base)/2. + sp.stats.entropy(q, m, base=base)/2.


# In[14]:


x = np.arange(-10,10,0.001)
p = norm.pdf(x, 4.3, 3.5)
q = norm.pdf(x, 4.1, 3.6)
print(p)
print(len(p))
plt.title('JSH')
print(jsd(p, q))
plt.plot(x, p, c='blue')
plt.plot(x, q, c='red')


# In[15]:


q = norm.pdf(x, 5, 4)
plt.title('JSH 2')
print(jsd(p, q))
plt.plot(x, p, c='blue')
plt.plot(x, q, c='red')

