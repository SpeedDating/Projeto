import scipy.stats as stats
import numpy as np
import pandas as pd

df = pd.read_csv("speeddating.csv", low_memory=False)
# print(df)

# ages_mean = np.mean(ages)
# print(ages_mean)

importance = pd.to_numeric(df[(df["importance_same_race"] != "?")]["importance_same_race"])
print(importance)

tset, pval = stats.ttest_1samp(importance, [5, 6, 7, 8, 9, 10])
print("p-values",pval)

if pval < 0.05:    # alpha value is 0.05 or 5%
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")