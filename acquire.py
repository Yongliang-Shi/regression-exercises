#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import env


# ### 1. Make a function named `get_titanic_data` that returns the titanic data from the codeup data science database as a pandas data frame. Obtain your data from the Codeup Data Science Database.

# In[ ]:


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[ ]:


def get_titanic_data():
    filename = 'titanic.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql('select * from passengers', get_connection('titanic_db'))
        df.to_csv(filename)
        return df


# ### 2. Make a function named `get_iris_data` that returns the data from the `iris_db` on the codeup data science database as a pandas data frame. The returned data frame should include the actual name of the species in addition to the `species_ids`. Obtain your data from the Codeup Data Science Database.

# In[9]:


def get_iris_data():
    filename = 'iris.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select * from measurements join species using(species_id)""", get_connection('iris_db'))
        df.to_csv(filename)
        return df


# In[ ]:

###################### Acquire Mall Customers Data ###########################

def get_mall_data():
    filename = 'mall_customers.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select * from customers""", get_connection('mall_customers'))
        df.to_csv(filename)
        return df