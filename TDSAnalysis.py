#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:47:00 2021

@author: semmerson
"""

import os, csv, pickle
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
#os.chdir('data')
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

ind0 = TDSdata[:,6] < 86
ind1 = (TDSdata[:,6] >= 86) & (TDSdata[:,6] < 111)
ind2 = (TDSdata[:,6] >= 111) & (TDSdata[:,6] < 136)
ind3 = (TDSdata[:,6] >= 136) & (TDSdata[:,6] < 166)
ind4 = TDSdata[:,6] >= 166
inds = np.vstack((ind0,ind1,ind2,ind3,ind4))
cols = ['#AFFFAF','#0FFF0F','#FAFA00','#EA0000','#FF00FF']

VR = 59.7
TDSH = 12.5
loc = 'Calera, AL'

X, Y, Z = np.mgrid[0:160:641j, 0:50:201j,50:220:171j]
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
if "vals" not in locals():
    try:  
        file = open("kde.pkl",'rb')
        vals = pickle.load(file)
        file.close()
    except:
        kde = sm.nonparametric.KDEMultivariate(TDSdata[:,np.array([1,4,6])].T,'ccc',bw=np.array([7.32, 2.54, 4.51]))
        vals = np.reshape(kde.pdf(positions),X.shape)
        filehandler = open("kde.pkl","wb")
        pickle.dump(vals,filehandler)
        filehandler.close()
#kde = stats.gaussian_kde(TDSdata[:,np.array([1,4,6])].T,bw_method=0.38)
#    vals = np.reshape(kde(positions),X.shape)

kss = np.nansum(vals,axis=2)
ks = vals/kss[:,:,np.newaxis]
exp = np.nansum(ks*np.linspace(50,220,171),axis=2)

expected = exp[int(VR*4),int(TDSH*4)]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
thresh = 0.15
p = 3e-3
marked = False

x = np.linspace(50,220,171)
y = vals[int(VR*4),int(TDSH*4),:]
y = y/np.sum(y)
ax.plot(x,y,color='black')
ind = np.where(x == 85)[0][0]
ind1 = np.where(x == 110)[0][0]
ax.fill_between(x[:ind+1],y[:ind+1],facecolor=cols[0],alpha=0.5)
if np.sum(y[:ind+1]) > thresh:
    xpos = np.mean(x[:ind+1][y[:ind+1] > p])
    ax.annotate('EF0',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color='#8FCF8F')
    ax.annotate('Sam Emmerson\n@ou_sams',(200,4.5e-3),alpha=0.3,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center')
    #ax.annotate(f'ML Windspeed: {x[np.argmax(y)]} mph',(200,12e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
    ax.annotate(f'PMM Windspeed: {np.round(np.dot(x,y/np.sum(y)),1)} mph',(200,10e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
    marked = True
ax.fill_between(x[ind:ind1+1],y[ind:ind1+1],facecolor=cols[1],alpha=0.5)
if np.sum(y[ind:ind1+1]) > thresh:
    xpos = np.mean(x[ind:ind1+1][y[ind:ind1+1] > p])
    ax.annotate('EF1',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color='#0FCF0F')
    ax.annotate('Sam Emmerson\n@ou_sams',(200,4.5e-3),alpha=0.3,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center')
    #ax.annotate(f'ML Windspeed: {x[np.argmax(y)]} mph',(200,12e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
    ax.annotate(f'PMM Windspeed: {np.round(np.dot(x,y/np.sum(y)),1)} mph',(200,10e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
    marked = True
ind = ind1
ind1 = np.where(x == 135)[0][0]
ax.fill_between(x[ind:ind1+1],y[ind:ind1+1],facecolor=cols[2],alpha=0.5)
if np.sum(y[ind:ind1+1]) > thresh:
    xpos = np.mean(x[ind:ind1+1][y[ind:ind1+1] > p])
    ax.annotate('EF2',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color='#CACA00',zorder=9)
ind = ind1
ind1 = np.where(x == 165)[0][0]
ax.fill_between(x[ind:ind1+1],y[ind:ind1+1],facecolor=cols[3],alpha=0.5)
if np.sum(y[ind:ind1+1]) > thresh:
    xpos = np.mean(x[ind:ind1+1][y[ind:ind1+1] > p])
    ax.annotate('EF3',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color=cols[3])
ind = ind1
ind1 = np.where(x == 200)[0][0]
ax.fill_between(x[ind:ind1+1],y[ind:ind1+1],facecolor=cols[4],alpha=0.5)
if np.sum(y[ind:ind1+1]) > thresh:
    xpos = np.mean(x[ind:ind1+1][y[ind:ind1+1] > p])
    ax.annotate('EF4',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color=cols[4])
ind = ind1
ind1 = np.where(x == 220)[0][0]
ax.fill_between(x[ind:ind1],y[ind:ind1],facecolor='#9B2FFF',alpha=0.5)
if np.sum(y[ind:ind1+1]) > thresh:
    xpos = np.mean(x[ind:ind1+1][y[ind:ind1+1] > p])
    ax.annotate('EF5',(xpos,1e-3),alpha=0.8,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center',color='#7B0FDF')
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ymax = 5e-2
ax.set_ylim(0,ymax)
#l1 = ax.axvline(x=x[np.argmax(y)],ymax=y[np.argmax(y)]/ymax,color=cols[3],ls=':')
l2 = ax.axvline(x=np.dot(x,y/np.sum(y)),ymax=y[int(np.rint(np.dot(x,y/np.sum(y)))-x[0]+1)]/ymax,color='#CACA00',ls='--')
ax.set_title(f'Intensity PDF for {np.round(TDSH,1)} kft TDS + {VR} kt Vrot')
ax.set_xlabel('Peak Wind Speed (mph)')
ax.set_ylabel('Probability')
#ax.legend([l1,l2],['ML','PMM'])
if not marked:
    ax.annotate('Sam Emmerson\n@ou_sams',(75,4.5e-3),alpha=0.3,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center')
    #ax.annotate(f'ML Windspeed: {x[np.argmax(y)]} mph',(75,12e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
    ax.annotate(f'Mean Windspeed: {np.round(np.dot(x,y/np.sum(y)),1)} mph',(75,10e-3),alpha=0.5,fontsize=14,weight='heavy',ha='center')
plt.tight_layout()
plt.savefig('TDSpdf.png')



# Ellipse plotting part
g = sns.JointGrid(TDSdata[:,1],TDSdata[:,4],height=7,xlim=(0,160),ylim=(0,40))
ellipses = []
for i in range(5):
    fd = TDSdata[inds[i,:],:]
    e = plot_point_cov(fd[:,np.array([1,4])],nstd=2,ax=g.ax_joint,alpha=0.3,color=cols[i],lw=1.5)
    sns.kdeplot(fd[:,1],ax=g.ax_marg_x,color=cols[i],shade=True,cut=0)
    sns.kdeplot(fd[:,4],ax=g.ax_marg_y,color=cols[i],vertical=True,shade=True,cut=0)
    ellipses.append(e)
g.ax_joint.set_xlabel('Vrot (kt)')
g.ax_joint.set_ylabel('TDS Height (kft)')
g.ax_joint.legend(ellipses,['EF0','EF1','EF2','EF3','EF4+'])
g.ax_joint.scatter(x=VR,y=TDSH,c='#AFAFAF',marker="v",s=75,edgecolors='black',zorder=2)
g.ax_joint.annotate(loc+' EFU',(VR+3,TDSH-0.6),fontsize=13,ha='left')
g.ax_joint.annotate('Sam Emmerson\n@ou_sams',(130,2),alpha=0.3,fontfamily='Tahoma',fontsize=16,weight='heavy',ha='center')
plt.tight_layout()
fig = plt.gcf()
fig.set_figheight(7)
fig.set_figwidth(11)
plt.savefig('TDSH.png')

obsv = np.array([VR,TDSH,expected])

# Analog finder
windows = np.array([10,5,30])
analogs = np.all(np.logical_and(TDSdata[:,np.array([1,4,6])] > obsv-windows,TDSdata[:,np.array([1,4,6])] < obsv+windows),axis=1)
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
scores = np.sum((((TDSdata[analogs,:][:,np.array([1,4,6])] - obsv))*np.array([0.45,0.5,0.05]))**2,axis=1)
sinds = np.argsort(scores)
ana_inds = np.where(analogs)[0]
ana_inds = ana_inds[sinds]
ana_tags = tagdata[analogs]
ana_tags = ana_tags[sinds]
print(f' Model Expected Windspeed: {np.round(np.dot(y,np.linspace(50,220,171)),2)} mph')
print(f" Best Match Windspeed: {TDSdata[ana_inds[0],6]} mph \n")

if len(ana_inds) < 5:
    n = len(ana_inds)
else:
    n = 5
print(f" Top {n} Analogs:")
for i in range(n):
    tag = ana_tags[i]
    if TDSdata[ana_inds[i],6] < 86:
        ef = 'EF0'
        col = cols[0]
    elif TDSdata[ana_inds[i],6] < 111:
        ef = 'EF1'
        col = cols[1]
    elif TDSdata[ana_inds[i],6] < 136:
        ef = 'EF2'
        col = cols[2]
    elif TDSdata[ana_inds[i],6] < 166:
        ef = 'EF3'
        col = cols[3]
    elif TDSdata[ana_inds[i],6] < 201:
        ef = 'EF4'
        col = cols[4]
    else:
        ef = 'EF5'
    print(f" #{i+1}: {int(TDSdata[ana_inds[i],6])} mph {ef} {tag[5]} County, {tag[4]} {tag[1]}/{tag[2]}/{tag[0]} @{str.rjust(tag[3],4,'0')}z")
          




