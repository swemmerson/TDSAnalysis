#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:47:00 2021

@author: semmerson
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import statsmodels.api as sm

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    # Wrapper function for drawing a confidence ellipse for a set of points
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    # Function for generating and plotting a confidence ellipse given 
    # a covariance matrix, position, and confidence level in standard deviations
    if ax is None:
        ax = plt.gca()
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

# Ugly data import
os.chdir('data')
data = []
header = ''
with open('TDSdata.csv') as f:
    reader = csv.reader(f,delimiter=',')
    for (n,row) in enumerate(reader):
        if n != 0:
            data.append(row)
        else:
            header = row
with open('TDSverf.csv') as f:
    reader = csv.reader(f,delimiter=',')
    for (n,row) in enumerate(reader):
        if n != 0:
            data.append(row)

TDS = np.empty(len(data))
TDSdata = np.zeros((len(data),9))
tags = []
for (n,row) in enumerate(data):            
    TDS[n] = bool(int(row[0]))
    tags.append(row[-1])
    if TDS[n]:
        TDSdata[n,:7] = np.array(row[1:8]).astype(float)
        TDSdata[n,7:9] = np.array(row[10:12]).astype(float)
    else:
        TDSdata[n,1:4] = np.array(row[2:5]).astype(float)
        TDSdata[n,6] = float(row[7])
        TDSdata[n,7:9] = np.array(row[10:12]).astype(float)

TDSdata = TDSdata[TDS.astype(bool),:]
tags = np.array(tags)
tags = tags[TDS.astype(bool)]
tagdata = []
for tag in tags:
    tagdata.append([int(tag[0:4]),int(tag[4:6]),int(tag[6:8]),int(tag[9:13]),tag[14:16],tag[16:-5],tag[-4:]])

tagdata = np.array(tagdata)
# renorm = False
# if renorm:
#     p1 = 3.24
#     p2 = 2.656
#     i1 = 6
#     TDSdata[:,3] = (np.log(TDSdata[:,3])+i1)**p2
#     TDSdata[:,4] = np.log(TDSdata[:,4])**p1
# zscores = (TDSdata-np.mean(TDSdata,axis=0))/np.std(TDSdata,axis=0)
# scores = np.prod(zscores[:,np.array([1,3,4])]*np.array([0.4,0.1,0.5]),axis=1)
# sinds = np.argsort(scores)[::-1]
# n = 10
# print(f'\n Top {n} Tornadoes by Radar Presentation: \n')
# for i in range(n):
#     tag = tagdata[sinds[i]]
#     if TDSdata[sinds[i],6] < 86:
#         ef = 'EF0'
#     elif TDSdata[sinds[i],6] < 111:
#         ef = 'EF1'
#     elif TDSdata[sinds[i],6] < 136:
#         ef = 'EF2'
#     elif TDSdata[sinds[i],6] < 165:
#         ef = 'EF3'
#     elif TDSdata[sinds[i],6] < 201:
#         ef = 'EF4'
#     else:
#         ef = 'EF5'
#     print(f" #{str.ljust(str(i+1)+':',3)} {int(TDSdata[sinds[i],6])} mph {ef} {tag[5]} County, {tag[4]} {tag[1]}/{tag[2]}/{tag[0]} @{str.rjust(tag[3],4,'0')}z")

ind0 = TDSdata[:,6] < 86
ind1 = (TDSdata[:,6] >= 86) & (TDSdata[:,6] < 111)
ind2 = (TDSdata[:,6] >= 111) & (TDSdata[:,6] < 136)
ind3 = (TDSdata[:,6] >= 136) & (TDSdata[:,6] < 166)
ind4 = TDSdata[:,6] >= 166
inds = np.vstack((ind0,ind1,ind2,ind3,ind4))
cols = ['#AFFFAF','#0FFF0F','#FAFA00','#EA0000','#FF00FF']

TDSH = 24
VR = 85.5

# Ellipse plotting part
g = sns.JointGrid(height=7,xlim=(0,160),ylim=(0,40))
ellipses = []
for i in range(5):
    fd = TDSdata[inds[i,:],:]
    e = plot_point_cov(fd[:,np.array([1,4])],nstd=2,ax=g.ax_joint,alpha=0.3,color=cols[i],lw=1.5)
    sns.kdeplot(x=fd[:,1],ax=g.ax_marg_x,common_norm=False,color=cols[i],fill=True,cut=0)
    sns.kdeplot(y=fd[:,4],ax=g.ax_marg_y,common_norm=False,color=cols[i],fill=True,cut=0)
    ellipses.append(e)
g.ax_joint.set_xlabel('Vrot (kt)')
g.ax_joint.set_ylabel('TDS Height (kft)')
g.ax_joint.legend(ellipses,['EF0','EF1','EF2','EF3','EF4+'])
g.ax_joint.scatter(x=VR,y=TDSH,c="#F7F7F7",marker="v",s=75,edgecolors='black')
g.ax_joint.annotate('Longwood, NC EFU',(VR+3,TDSH-0.4))
g.ax_joint.annotate('Sam Emmerson\n@ou_sams',(130,2),alpha=0.3,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center')
plt.tight_layout()

obsv = np.array([VR,TDSH])

# Analog finder
windows = np.array([5,2])
analogs = np.all(np.logical_and(TDSdata[:,np.array([1,4])] > obsv-windows,TDSdata[:,np.array([1,4])] < obsv+windows),axis=1)
print(f"\n TDS Analogs for a TDS height of {TDSH} kft and a Vrot of {VR} kt:")
print(f" Number of Analogs: {np.sum(analogs)} \n")
print(f" % EF0:  {np.round(100*np.sum(TDSdata[analogs,:][:,6] < 86)/np.sum(analogs),1)}%")
print(f" % EF1:  {np.round(100*np.sum((TDSdata[analogs,:][:,6] >= 86) & (TDSdata[analogs,:][:,6] < 111))/np.sum(analogs),1)}%")
print(f" % EF2:  {np.round(100*np.sum((TDSdata[analogs,:][:,6] >= 111) & (TDSdata[analogs,:][:,6] < 136))/np.sum(analogs),1)}%")
print(f" % EF3:  {np.round(100*np.sum((TDSdata[analogs,:][:,6] >= 136) & (TDSdata[analogs,:][:,6] < 166))/np.sum(analogs),1)}%")
print(f" % EF4+: {np.round(100*np.sum(TDSdata[analogs,:][:,6] >= 166)/np.sum(analogs),1)}%")
print("\n Min:        25th:       Median:     Mean:       75th:       Max:")
print(f" {np.min(TDSdata[analogs,6])} \
      {np.round(np.percentile(TDSdata[analogs,6],25),1)} \
      {np.round(np.percentile(TDSdata[analogs,6],50),1)} \
      {np.round(np.mean(TDSdata[analogs,6]),1)} \
      {np.round(np.percentile(TDSdata[analogs,6],75),1)} \
      {np.max(TDSdata[analogs,6])}\n")
scores = np.sum(((TDSdata[analogs,:][:,np.array([1,4])] - obsv)**2)*np.array([0.45,0.55]),axis=1)
sinds = np.argsort(scores)
ana_inds = np.where(analogs)[0]
ana_inds = ana_inds[sinds]
ana_tags = tagdata[analogs]
ana_tags = ana_tags[sinds]
print(f" Best Match Windspeed: {TDSdata[ana_inds[0],6]} mph ")

if len(ana_inds) < 5:
    n = len(ana_inds)
else:
    n = 5
print(f" Top {n} Analogs:")
for i in range(n):
    tag = ana_tags[i]
    if TDSdata[ana_inds[i],6] < 86:
        ef = 'EF0'
    elif TDSdata[ana_inds[i],6] < 111:
        ef = 'EF1'
    elif TDSdata[ana_inds[i],6] < 136:
        ef = 'EF2'
    elif TDSdata[ana_inds[i],6] < 166:
        ef = 'EF3'
    elif TDSdata[ana_inds[i],6] < 201:
        ef = 'EF4'
    else:
        ef = 'EF5'
    print(f" #{i+1}: {int(TDSdata[ana_inds[i],6])} mph {ef} {tag[5]} County, {tag[4]} {tag[1]}/{tag[2]}/{tag[0]} @{str.rjust(tag[3],4,'0')}z")
    
#lm = reg_m(TDSdata[:,6],TDSdata[:,3:5].T)
