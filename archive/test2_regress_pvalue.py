# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:52:18 2020

@author: current
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

k=0
idx=0
while idx < 1:
    rlist1 = []
    rlist2 = []
    for i in range(0,42): 
        n1 = random.uniform(1.0,1.1)
        rlist1.append(n1)
    for i in range(0,42): 
        n2 = random.uniform(1.0,1.1)
        rlist2.append(n2)
        
    r = stats.linregress(rlist1, rlist2)
    if r.pvalue < 0.1:
        idx=1
    k += 1
    
print(k)

print(np.corrcoef(rlist1,rlist2)[0,1])

x = np.sort(rlist1)
y = r.slope*x + r.intercept

fig=plt.figure(figsize=(4.5,4))
plt.plot(rlist1, rlist2,'go', x, y, 'r')



