import pandas as pd

df = pd.read_csv('speeddating.csv', low_memory=False, encoding='utf-8')

df = df.drop('interests_correlate',axis=1)
df = df.drop('d_interests_correlate',axis=1)
df = df.drop('field',axis=1)
df = df.drop('d_d_age',axis=1)

df = df.replace("'Asian/Pacific Islander/Asian-American'",1)
df = df.replace("'Latino/Hispanic American'",2)
df = df.replace("Other",3)
df = df.replace("'Black/African American'",4)
df = df.replace("European/Caucasian-American",5)

df = df.mask(df.eq('?')).dropna()

df = df.replace("female",1)
df = df.replace("male",2)

df = df.replace('[0-1]', 1)
df = df.replace('[2-5]', 2)
df = df.replace('[6-10]', 3)

df = df.replace('[0-15]', 1)
df = df.replace('[16-20]', 2)
df = df.replace('[21-100]', 3)

df = df.replace('[0-5]', 1)
df = df.replace('[6-8]', 2)
df = df.replace('[9-10]', 3)

df = df.replace('[0-4]', 1)
df = df.replace('[5-6]', 2)
df = df.replace('[7-10]', 3)

df = df.replace('[0-3]', 1)
df = df.replace('[4-9]', 2)
df = df.replace('[10-20]', 3)

df = df.replace('[0-2]', 1)
df = df.replace('[3-5]', 2)
df = df.replace('[5-18]', 3)