from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from preparecsv import df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_positive = df.loc[(df['decision_o']) == 1]
df_negative = df.loc[(df['decision_o']) == 0]

df_negative, df_negative_rest = train_test_split(df_negative, test_size = (1 - df_positive.shape[0]/df_negative.shape[0]), random_state = 1)

print(df_positive.shape[0])
print(df_negative.shape[0])
print(df_positive.shape[0]/df_negative.shape[0])
print(df_negative.shape[0] + df_negative_rest.shape[0])

df = pd.concat([df_positive, df_negative], ignore_index=True)

print(df.shape[0])

training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

test_set = pd.concat([test_set, df_negative_rest])

test = test_set.copy()
treino = training_set.copy()

yTeste = test_set[['decision_o']]
yTreino = training_set[['decision_o']]

del training_set['decision_o']
del test_set['decision_o']

# del training_set['decision']
# del test_set['decision']

del training_set['match']
del test_set['match']

# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
classifier = SVC(kernel='linear', C=0.15, random_state = 1)
classifier.fit(training_set, yTreino)

Y_pred = classifier.predict(test_set)

test_set["Predictions"] = Y_pred

cm = confusion_matrix(yTeste,Y_pred)
accuracy = float(cm.diagonal().sum())/len(yTeste) 
print("\n Acuracia do dataframe para o dataset : ", accuracy)


# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()

def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(50, 15))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=90, ha='right')
 plt.savefig('image.png')

# f_importances(classifier.coef_, features_names)
# del test_set["Predictions"]
# features_names = test_set.columns
# print(len(classifier.coef_.ravel()))
# print(len(test_set.columns))
# plot_coefficients(classifier, features_names, 58)