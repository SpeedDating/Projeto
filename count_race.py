import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("speeddating.csv", low_memory=False)

race = df.set_index(["race", "wave"]).count(level="race")
race = race.sort_values("age")

format = lambda x: x

race = race["age"].map(format)

labels = list(race.keys())
sizes = list(race.values)

plt.rcParams['font.size'] = 6.0

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, startangle=90, autopct="%1.2f%%")
ax1.axis("equal")
plt.savefig("race_percent.png")