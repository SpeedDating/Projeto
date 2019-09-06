import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("speeddating.csv", low_memory=False)

gender = df.set_index(["gender", "wave"]).count(level="gender")

num_women = gender["age"]["female"]
num_men = gender["age"]["male"]

labels = "Homens", "Mulheres"
sizes = [num_men, num_women]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, startangle=90, autopct="%1.2f%%")
ax1.axis("equal")
plt.savefig("gender_percent.png")