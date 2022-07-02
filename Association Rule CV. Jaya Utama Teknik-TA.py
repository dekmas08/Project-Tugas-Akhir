#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import mlxtend as mlxtend
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[2]:


df = pd.read_excel('olah.xlsx')
df


# In[3]:


barang = df['Item'].value_counts()
barang


# In[4]:


sns.countplot(x = 'Item', data = df, order = df['Item'].value_counts().iloc[:90].index)
plt.xticks(rotation=90)  


# # Preprocessing Data
# - menghapus atribut yang tidak digunakan
# - menghapus data yang kosong
# - transformasi data ke bentuk tabular 1 atau 0

# In[5]:


data=df.drop(['Tanggal','Hotel','Qty','Satuan','Harga Jual',
              'Total Jual','Jumlah','Harga Beli','Total Harga Beli',
              'Ongkos Angkut','Profit','%'],axis=1)
data


# In[6]:


data.isna().sum()


# In[7]:


data.dropna(axis=0, subset=['Item'], inplace=True)
data


# In[8]:


dataset = data.groupby(['No Do','Item']).size().reset_index(name='count')
basket = (dataset.groupby(['No Do', 'Item'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('No Do'))
basket


# In[9]:


def encode_units(x) :
    if x <= 0:
        return 0
    if x >=1 :
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets


# In[10]:


basket_filter = basket_sets[(basket_sets > 0).sum(axis=1) >=2]
basket_filter


# # Data Mining

# In[11]:


frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[12]:


# Create the rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.65)
hasil = rules[['antecedents','consequents','support','confidence']]
hasil


# In[13]:


hasil = rules[(rules['confidence'] >=0.65) &
           (rules['lift'] >1)]
apr_result = hasil.sort_values(by='lift', ascending=False)
apr_result[['antecedents','consequents','support','confidence','lift']]

