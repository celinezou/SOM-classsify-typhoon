# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:05:19 2017

@author: Zou_S.L
"""

import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection
import numpy as np
import matplotlib.cm as cm


m = 3   # The height
n = 3   # The width

# Some maths regarding hexagon geometry
d = 30
s = d/(2*np.cos(np.pi/3))
h = s*(1+2*np.sin(np.pi/3))
r = d/2
area = 3*np.sqrt(3)*s**2/2

# The center coordinates of the hexagons are calculated.
x1 = np.array([d*x for x in range(2*n-1)])
x2 = x1 + r
x3 = x2 + r
y = np.array([h*x for x in range(2*m-1)])
c = []

for i in range(2*m-1):
    if i%4 == 0:
        c += [[x,y[i]] for x in x1]
    if (i-1)%2 == 0:
        c += [[x,y[i]] for x in x2]
    if (i-2)%4 == 0:
        c += [[x,y[i]] for x in x3]
c = np.array(c)

# The color of the hexagons
d_matrix = np.zeros(3*3)

# Creating the figure
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)

# The collection
coll = RegularPolyCollection(
    numsides=6,  # a hexagon
    rotation=0,
    sizes=(area,),
    edgecolors = (0, 0, 0, 1),
    array= d_matrix,
    cmap = cm.gray_r,
    offsets = c,
    transOffset = ax.transData,
)

ax.add_collection(coll, autolim=True)

ax.autoscale_view()
plt.show()