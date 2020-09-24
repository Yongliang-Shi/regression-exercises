import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_telco_data():
    filename = 'telco_spend.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select customer_id, monthly_charges, tenure, total_charges
                    from customers
                    join contract_types using(contract_type_id)
                    where contract_type = 'Two year'""", get_connection('telco_churn'))
        df.to_csv(filename)
        return df
    
def prep_telco():
    df = get_telco_data()
    boolean = df.total_charges.str.isspace()
    df = df[-boolean]
    df['total_charges'] = df.total_charges.apply(lambda i: float(i))
    df.reset_index(drop=True, inplace=True)
    train_validate, test = train_test_split(df, test_size=.15, random_state=442)
    train, validate = train_test_split(train_validate, test_size=.175, random_state=442)
    return train, validate, test