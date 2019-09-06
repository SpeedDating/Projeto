import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("speeddating.csv", low_memory=False)

age = df.set_index(["age", "wave"]).count(level="age")

format = lambda x: x

age = age["gender"].map(format)

labels = list(age.keys())
sizes = list(age.values)
index = np.arange(len(labels))

plt.bar(index, sizes)
plt.xlabel('Idade', fontsize=10)
plt.ylabel('Número de Pessoas', fontsize=7)
plt.xticks(index, labels, fontsize=7, rotation=0)
plt.title('Idade dos Participante')
plt.savefig('age_percent.png', bbox_inches='tight')