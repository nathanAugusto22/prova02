# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

base_car = pd.read_csv('car_data.txt')
base_car.describe()

#verificar se tem linha nula;
base_car.loc[pd.isnull(base_car['pcompra'])] 
base_car.loc[pd.isnull(base_car['pmanutencao'])]
base_car.loc[pd.isnull(base_car['nportas'])]
base_car.loc[pd.isnull(base_car['nlugares'])]
base_car.loc[pd.isnull(base_car['tmala'])]
base_car.loc[pd.isnull(base_car['seguranca'])]
base_car.loc[pd.isnull(base_car['aceitacao'])]

#divisao da base de dados em atributos previsores e a classe;
previsores = base_car.iloc[:, 0:6].values
classe =  base_car.iloc[:, 6].values

#transformando variaveis categorica em numericas
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 4] = labelencoder_previsores.fit_transform(previsores[:, 4])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])



#dividindo as bases em treinos e testes
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#metodo para predicao
def fit_and_predict(nome, modelo, previsores_treinamento, classe_treinamento):
  modelo.fit(previsores_treinamento, classe_treinamento)
  result = modelo.predict(previsores_treinamento)

  correct = 0
  size = len(classe_treinamento)
  for i in range(size):
    if classe_treinamento[i] == result[i]:
      correct += 1

  print('%s: %.2f%%' %(nome, (correct*100/size)))


# k-fold
def k_fold(nome, modelo, previsores_treinamento, classe_treinamento, k):
  scores = cross_val_score(modelo, previsores_treinamento, classe_treinamento, cv = k)
  print("%s %d-fold: %.2f%%" %(nome, k, (np.mean(scores)*100.0)))

modeloMultinomial = MultinomialNB()
modeloAdaBoost = AdaBoostClassifier()
modeloRandomForest = RandomForestClassifier()
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
modeloNaive_Bayes_GaussianNB = GaussianNB()

fit_and_predict("MultinomialNB", modeloMultinomial, previsores_treinamento, classe_treinamento)
fit_and_predict("AdaBoostClassifier", modeloAdaBoost, previsores_treinamento, classe_treinamento)
fit_and_predict("RandomForestClassifier", modeloRandomForest, previsores_treinamento, classe_treinamento)
fit_and_predict("OneVsRestClassifier", modeloOneVsRest, previsores_treinamento, classe_treinamento)
fit_and_predict("OneVsOneClassifier", modeloOneVsOne, previsores_treinamento, classe_treinamento)
fit_and_predict("GaussianNB", modeloNaive_Bayes_GaussianNB, previsores_treinamento, classe_treinamento)
k = 6

print()

k_fold("MultinomialNB", modeloMultinomial, previsores_treinamento, classe_treinamento, k)
k_fold("AdaBoostClassifier", modeloAdaBoost, previsores_treinamento, classe_treinamento, k)
k_fold("RandomForestClassifier", modeloRandomForest, previsores_treinamento, classe_treinamento, k)
k_fold("OneVsRestClassifier", modeloOneVsRest, previsores_treinamento, classe_treinamento, k)
k_fold("OneVsOneClassifier", modeloOneVsOne, previsores_treinamento, classe_treinamento, k)
k_fold("GaussianNB", modeloNaive_Bayes_GaussianNB, previsores_treinamento, classe_treinamento, k)

#gerando graficos
sns.set(style="whitegrid", color_codes=True)
%matplotlib inline
sns.countplot(x = "aceitacao", data = base_car, palette = "Greens_d");
sns.factorplot(x="aceitacao", y="nlugares", data=base_car, kind="bar");

#codigos

#prevendo com  naive_bayes
#classificador = GaussianNB() #criando classificador
#classificador.fit(previsores_treinamento, classe_treinamento) #gerando tabelas de probabilidade
#previsoes_GaussianNB = classificador.predict(previsores_teste) #fazendo predict

# comparando classe de teste com o que o algoritmo encontrou
#from sklearn.metrics import accuracy_score, confusion_matrix
#precisao = accuracy_score(classe_teste, previsoes_GaussianNB)
#matriz = confusion_matrix(classe_teste,previsoes_GaussianNB)
#print(precisao*100)


#sns.pairplot(base_car, hue='rating');
#sns.barplot(x="aceitacao", y="seguranca", data=base_car);
#sns.barplot(x="aceitacao", y="nlugares", data=base_car);

#escolanamento
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)

#n_estimators=1295
#learning_rate=1
#previsoes_AdaBoostClassifier = AdaBoostClassifier(
#    base_estimator=previsores_treinamento,
#    learning_rate=learning_rate,
#    n_estimators=n_estimators,
#    algorithm="SAMME")
#previsoes_AdaBoostClassifier.fit(previsores_treinamento, classe_treinamento)
#print(previsoes_AdaBoostClassifier)










