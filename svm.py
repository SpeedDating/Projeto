from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from preparecsv import df


training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)

test = test_set.copy()
treino = training_set.copy()

yTeste = test_set[['decision_o']]
yTreino = training_set[['decision_o']]

del training_set['decision_o']
del test_set['decision_o']

# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
classifier = SVC(kernel='linear', C=0.08, random_state = 1)
classifier.fit(training_set, yTreino)

Y_pred = classifier.predict(test_set)

test_set["Predictions"] = Y_pred

cm = confusion_matrix(yTeste,Y_pred)
accuracy = float(cm.diagonal().sum())/len(yTeste) 
print("\n Acuracia do dataframe para o dataset : ", accuracy)