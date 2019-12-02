import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('speeddating.csv', low_memory=False, encoding='utf-8')

df = df[['gender', 'age']].mask(df.eq('?')).dropna()

def convert_age(age):
    age = int(age)
    if age <= 20:
        return '0-20'

    if age <= 25:
        return '21-25'

    if age <= 30:
        return '26-30'

    if age <= 40:
        return '31-40'

    if age > 40:
        return '41+'

df['age'] = df['age'].apply(convert_age)
df['age'] = df['age'].astype(str)
df['gender'] = df['gender'].astype(str)


# print(df['age'], df['gender'])

man_ages = [df.loc[(df['age'] == '0-20') & (df['gender'] == 'male')].shape[0], df.loc[(df['age'] == '21-25') & (df['gender'] == 'male')].shape[0], df.loc[(df['age'] == '26-30') & (df['gender'] == 'male')].shape[0], df.loc[(df['age'] == '31-40') & (df['gender'] == 'male')].shape[0], df.loc[(df['age'] == '41+') & (df['gender'] == 'male')].shape[0]]
woman_ages = [df.loc[(df['age'] == '0-20') & (df['gender'] == 'female')].shape[0], df.loc[(df['age'] == '21-25') & (df['gender'] == 'female')].shape[0], df.loc[(df['age'] == '26-30') & (df['gender'] == 'female')].shape[0], df.loc[(df['age'] == '31-40') & (df['gender'] == 'female')].shape[0], df.loc[(df['age'] == '41+') & (df['gender'] == 'female')].shape[0]]

print(man_ages, woman_ages)

print(df.shape[0])
# print(df.head())

fig, ax = plt.subplots()

ind = np.arange(5)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, man_ages, width, bottom=0, color='#000000')

p2 = ax.bar(ind + width, woman_ages, width, bottom=0, color='#a9a9a9')

ax.set_title('Número de pessoas por idade e gênero')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('0-20', '21-25', '26-30', '31-40', '41+'))
ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])

ax.legend((p1[0], p2[0]), ('Homens', 'Mulheres'))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(p1)
autolabel(p2)

ax.autoscale_view()

plt.savefig('age_graph4.png')