import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Nazwa pliku z którego pobierane są dane
name = 'capture20110810.csv'
# Wczytanie pliku
df = pd.read_csv(name)
# Zamiana oznaczeń Normal na 0 a Attack na 1
df['label'].replace('Normal', 0, inplace=True)
df['label'].replace('Attack', 1, inplace=True)
# Zdefinowanie dla których kolumn ma być przeprowadzona analiza
X = df[['n_tcp','n_udp', 'n_icmp', 'background_flow_count', 'normal_flow_count', 'n_conn']].values
Y = df['label'].values.ravel()
# Tasowanie danych
X, Y = shuffle(X, Y, random_state = 415)
# Stworzenie danych testowych i ćwiczeniowych (20% bazy do testów)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20 ,random_state = 101)

# Zdefiniowanie funkcji dla regresji logistycznej
def LogRegression(X_train,y_train,X_test, Y_test):
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    # predictions = logmodel.predict(X_test)
    # Zapisanie modelu na dysku
    filename = 'LogisticRegression_{}.sav'.format(name[:-4])
    joblib.dump(logmodel, filename)
    # Wczytanie modelu
    loaded_model = joblib.load(filename)
    # Określenie skuteczności
    result = loaded_model.score(X_test, Y_test)
    print(result)

# Zdefiniowanie funkcji dla Support Vector Machine
def SupportVectorMachines(X_train,y_train,X_test, Y_test):
    model = SVC()
    model.fit(X_train,y_train)
    # predictions = model.predict(X_test)
    # Zapisanie modelu na dysku
    filename = 'SupportVectorMachines_{}.sav'.format(name[:-4])
    joblib.dump(model, filename)
    # Wczytanie modelu
    loaded_model = joblib.load(filename)
    # Określenie skuteczności
    result = loaded_model.score(X_test, Y_test)
    print(result)
    # Sprawdzenie czy z innymi współczynikami wynik będzie lepszy
    print("*"*100)
    print('Sprawdzenie parametrów siatki:')
    # Zdefiniowanie siatki parametrów
    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
    grid.fit(X_train,y_train)
    # Wypisanie najlepszych parametrów
    print('Najlepszy parametr')
    print(grid.best_params_)
    print('Najlepszy estymator')
    print(grid.best_estimator_)
    # grid_predictions = grid.predict(X_test)
    filename = 'SupportVectorMachines_grid_{}.sav'.format(name[:-4])
    joblib.dump(grid, filename)
    # Wczytanie modelu
    loaded_model = joblib.load(filename)
    # Określenie skuteczności
    result = loaded_model.score(X_test, Y_test)
    print(result)

# Zdefiniowanie funkcji dla drzewa decyzyjnego
def DTreeClassifier(X_train,y_train,X_test, Y_test):
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    # predictions = dtree.predict(X_test)
    # Zapisanie modelu na dysku
    filename = 'DecisionTreeClassifier_{}.sav'.format(name[:-4])
    joblib.dump(dtree, filename)
    # Wczytanie modelu
    loaded_model = joblib.load(filename)
    # Określenie skuteczności
    result = loaded_model.score(X_test, Y_test)
    print(result)

# Zdefiniowanie funkcji dla Random Forest Classifier
def RanForestClassifier(X_train,y_train,X_test, Y_test):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    # rfc_pred = rfc.predict(X_test)
    # Zapisanie modelu na dysku
    filename = 'RandomForestClassifier_{}.sav'.format(name[:-4])
    joblib.dump(rfc, filename)
    # Wczytanie modelu
    loaded_model = joblib.load(filename)
    # Określenie skuteczności
    result = loaded_model.score(X_test, Y_test)
    print(result)

LogRegression(x_train, y_train, x_test, y_test)
SupportVectorMachines(x_train, y_train, x_test, y_test)
DTreeClassifier(x_train, y_train, x_test, y_test)
RanForestClassifier(x_train, y_train, x_test, y_test)
