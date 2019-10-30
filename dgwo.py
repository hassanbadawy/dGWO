# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%% ---------------------------------Libraries---------------------------------
from __future__ import division
from __future__ import print_function
# import torch
import textdistance as td
import pdb
import glob
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import copy as cp
from collections import deque
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tqdm import tqdm




# %%
class dgwo():
    def __init__(self, pool, indv_size, pop_size = 10, target='', no_gen = 1):
        self.pool = pool
        self.indv_size = indv_size
        self.pop_size = pop_size
        self.target = target
        self.indv = []
        self.pop = []
        self.no_gen = no_gen

    def individual(self,):
        self.indv = random.sample(self.pool,self.indv_size)
        return self.indv 

    def populate(self,):
        for n in range(self.pop_size):
            self.pop.append(self.individual())
        return self.pop

    def objf(self, indv_arg):
        '''
        This function check if the given two text is the same or not using the levenshtein normalized distance,
        if the to text is identical, return 0.0 else return 0.0> = number <= 1.0. 
        '''
        return td.levenshtein.normalized_distance(''.join(indv_arg), self.target)

    def mutate(self, indv_arg, mutation_rate):
        '''
        this function return the mutation for chiled by mutation rate as a factor.
        '''
        mutation_value = int(round(mutation_rate * len(indv_arg)))
        mutation_pool = random.sample(self.pool, mutation_value)
        indv = cp.deepcopy(indv_arg)
        for i in range(mutation_value):
            mutation_choice = random.choice(mutation_pool)
            mutation_pool.remove(mutation_choice)
            idx = random.randint(0,len(indv)-1)
            indv[idx] = mutation_choice
        return indv

    def crossover2(self, indv1, indv2, crossover_rate):
        '''
        this function return the cross over between two agents defined by crossover rate.
        '''
        crossover_value = int(round(crossover_rate * len(indv1),0))
        flag = np.random.randint(0,2)
        if flag==1:
            child = indv1[:crossover_value] + indv2[crossover_value:]
            
        else:
            child = indv2[:crossover_value] + indv1[crossover_value:]
        return child

    def crossover3(self, indv1, indv2, indv3, crossover_rate):
        '''
        this function return the cross over between three indvs defined by crossover rate.
        '''

        crossover_value = int(round(crossover_rate * len(indv1),0))
        flag = np.random.randint(0,3)
        if flag == 0: 
            child = indv1[:crossover_value] \
            + indv2[crossover_value:crossover_value*2] \
            + indv3[crossover_value*2:]
        elif flag == 1: 
            child = indv1[:crossover_value] \
            + indv3[crossover_value:crossover_value*2] \
            +indv2[crossover_value*2:]
        else: 
            child = indv3[:crossover_value] \
            + indv2[crossover_value:crossover_value*2] \
            + indv1[crossover_value*2:]
        return child
    def get_AC(self, a):
        r1=random.random() # if r1>0.5: A1=[0:2] -->converge --> crossover
        r2=random.random() # if r1<0.5: A1=[-2:0] --> Diverge --> Mutation
        A=2*a*r1-a; # Equation (3.3)
        C=2*r2; # Equation (3.4)     
        return A,C 

    def run(self,):

        Convergence_curve = []        
        Last_Alpha_score =float("inf")
        #-------------------------------------------------
        FILE_PATH = 'logs.csv'
        positions = self.populate()
        #--------------------------------------------------    
        
        for no_gen_i in tqdm(range(self.no_gen)):
            wolves_df = {'fitness':[], 'position':[]}
            for agent in positions:
                wolves_df['fitness'].append(self.objf(agent))
                wolves_df['position'].append(agent)

            wolves_df = pd.DataFrame(data=wolves_df).sort_values(by=['fitness']).reset_index(drop=True)
            Alpha_score, Alpha_pos = wolves_df.loc[0, 'fitness'], wolves_df.loc[0, 'position']
            Beta_score, Beta_pos   = wolves_df.loc[1, 'fitness'], wolves_df.loc[1, 'position']
            Delta_score, Delta_pos = wolves_df.loc[2, 'fitness'], wolves_df.loc[2, 'position']
            if Alpha_score<Last_Alpha_score:
                Last_Alpha_score = Alpha_score
            Convergence_curve.append(Last_Alpha_score)  
 
            #a=2-l*((2)/gen_no); # a decreases linearly fron 2 to 0 --> a =
            a= 2*(1-(no_gen_i/self.no_gen)**2)
            positions = [Alpha_pos, Beta_pos, Delta_pos]

            for i in range(3, self.pop_size):               
                X1,X2,X3 = [], [], []
        
                agent_fitness = wolves_df.loc[i, 'fitness']
                B0 = wolves_df.loc[i, 'position']

                A1, C1 = self.get_AC(a)
                A2, C2 = self.get_AC(a)
                A3, C3 = self.get_AC(a)

                D_alpha = abs((agent_fitness - C1*Alpha_score)/agent_fitness) 
                D_beta = abs((agent_fitness - C2*Beta_score)/agent_fitness)
                D_delta = abs((agent_fitness - C3*Delta_score)/agent_fitness)
    
                if abs(A1)>=1:
                    #Exploration
                    X1 = self.mutate(B0, D_alpha)
                else:
                    #Exploitation
                    X1 = self.crossover2(Alpha_pos, B0, D_alpha)

                if abs(A2)>=1:
                    #Exploration
                    X2 = self.mutate(B0, D_beta)
                else:
                    #Exploitation
                    X2 = self.crossover2(Beta_pos, B0, D_beta)

                if abs(A3)>=1:
                    #Exploration
                    X3 = self.mutate(B0, D_delta)
                else:
                    #Exploitation
                    X3 = self.crossover2(Delta_pos, B0, D_delta)
                X = self.crossover3(X1, X2, X3, 0.3)
                positions.append(X)   

        return Convergence_curve, Alpha_pos
#%%         
if __name__ == "__main__":
    pool = list('abcdefghijklmnopqrstuvwxyz ')
    target = 'to be or not to be'
    opt = dgwo(pool=pool, indv_size=len(target), pop_size = 10, 
    target = target, no_gen=50)
    conv, pos = opt.run()
    plt.plot(conv)
    plt.ylim(0, 1)
    plt.show()
    print(''.join(pos))


# %%

# %%
