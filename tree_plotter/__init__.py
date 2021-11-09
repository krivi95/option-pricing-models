# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:12:03 2021

@author: ztche
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_lattice(tree, style='ko-'):
    """
    plots stock lattice on matplotlib from chronological binary tree

    """
    vals1 = {}
    for i, v in enumerate(tree):
        vals1[i] = v
    
    df1 = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in vals1.items()]))
    
    vals2 = {}
    for i, v in enumerate(tree):
        vals2[i] = reversed(v)
    
    df2 = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in vals2.items()]))
    
    plt.plot(df1.transpose(), style)
    plt.plot(df2.transpose(), style)
    plt.xlabel("Time Period")
    plt.ylabel("")
    
    S = tree[0][0]
    plt.annotate(S, (0, S), textcoords="offset points",
                 xytext=(0,-15), ha='center')

def plot_option_lattice(tree, style='bo-'):
    """
    plots options lattice from reverse-chronological binary tree
    
    """
    vals1 = {}
    for i, v in enumerate(reversed(tree)):
        vals1[i] = v
    
    df1 = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in vals1.items()]))
    
    vals2 = {}
    for i, v in enumerate(reversed(tree)):
        vals2[i] = reversed(v)
    
    df2 = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in vals2.items()]))
    
    plt.plot(df1.transpose(), style)
    plt.plot(df2.transpose(), style)
    
    V0 = tree[-1][-1]
    plt.annotate(round(V0, 2), (0, V0), textcoords="offset points",
                 xytext=(0,-15), ha='center')