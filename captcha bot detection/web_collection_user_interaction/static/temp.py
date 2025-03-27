#splitting the dataset into 2 parts the 1st 501 and last 500

import pandas as pd
import numpy as np

df=pd.read_csv('dataset\captcha_dataset.csv')

df1=df.iloc[:501,:]
df2=df.iloc[501:,:]

df1.to_csv('dataset\captcha_bot_intelligent.csv',index=False)
df2.to_csv('dataset\captcha_bot_basic.csv',index=False)

print(df1.shape)
print(df2.shape)