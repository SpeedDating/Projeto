import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests

df = pd.read_csv("speeddating.csv", low_memory=False,sep=",",header=0,usecols=["importance_same_race","race"])
df = df[(df["importance_same_race"] != "?")]


df_europeu = df[(df['race'] == "European/Caucasian-American") ]
print(df_europeu.race.count())
df = df[(df['race'] == "'Asian/Pacific Islander/Asian-American'") ]
print(df.race.count())


df['importance_same_race'] = df['importance_same_race'].astype('int64')
df_europeu['importance_same_race'] = df_europeu['importance_same_race'].astype('int64')

df = df[0:1500]
df_europeu = df_europeu[0:1500]

ztest, pval1 = stests.ztest (df_europeu['importance_same_race'], x2 = df['importance_same_race'],value=1,alternative='two-sided') 
print(pval1)
if pval1 < 0.05: 
    print ( "Hipótese Falsa") 
else: 
    print ("Hipótese verdadeira")