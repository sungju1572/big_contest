# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:40:25 2022

@author: user
"""




#clustering 


import pandas as pd

loan_data = pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/loan_result.csv')

log_data =pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/log_data.csv')

user_spec =pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/user_spec.csv')


len(loan_data['application_id'].unique())


len(user_spec['application_id'].unique())

key_value = log_data['user_id'].unique()

sna_data = user_spec[user_spec['user_id'].isin(key_value)]



sna_data.columns

sna_data['income_type']

sna_1 = sna_data[sna_data['income_type'] =='PRACTITIONER']

sna_1 = sna_1[['user_id','credit_score']]


sna_1 = sna_1.drop_duplicates(['user_id'])


pr_sna = sna_1[0:150]

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


g = nx.Graph()
g = nx.from_pandas_edgelist(pr_sna, source = 'user_id', target = 'credit_score')
print(nx.info(g))

plt.figure(figsize=(20, 10))
pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g,pos, node_size = 25, node_color = 'blue')
plt.show()