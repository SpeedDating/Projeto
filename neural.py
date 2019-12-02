from preparecsv import df
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import pandas as pd
import numpy as np

df_positive = df.loc[(df['decision_o']) == 1]
df_negative = df.loc[(df['decision_o']) == 0]

df_negative, df_negative_rest = train_test_split(df_negative, test_size = (1 - df_positive.shape[0]/df_negative.shape[0]), random_state = 1)

df = pd.concat([df_positive, df_negative], ignore_index=True)

training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

test_set = pd.concat([test_set, df_negative_rest])

test = test_set.copy()
treino = training_set.copy()

yTeste = test_set[['decision_o']]
yTreino = training_set[['decision_o']]

del training_set['decision_o']
del test_set['decision_o']

del training_set['match']
del test_set['match']

mlp_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1024, 128), random_state=1, max_iter=150)

mlp_model.fit(training_set, yTreino)

y_pred = mlp_model.predict(test_set)

test_set['Predictions'] = y_pred

print('Mean Absolute Error:', metrics.mean_absolute_error(yTeste, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yTeste, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yTeste, y_pred)))

accuracy_s = accuracy_score(yTeste, y_pred)
print('Accuracy Score:', accuracy_s)