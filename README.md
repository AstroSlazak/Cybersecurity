# Cybersecurity
Projekt przejściowy - cybersecurity

[Baza danych](https://mcfp.weebly.com/the-ctu-13-dataset-a-labeled-dataset-with-botnet-normal-and-background-traffic.html)

### Wykresy danych dla pierwszego przypadku:
* Heatmap:
![alt text](http://imagizer.imageshack.us/a/img924/7729/Da76Qq.png )
* Pairplot dla 3 wybranych kolumn:
![alt text](http://imageshack.com/a/img924/6415/WQo4WO.png)


### Przy użyciu scikit-learn skuteczność wykrywania anomali wynosi:
* #### Logistic Regression = 1.0
* #### Support Vector Machines = 0.846380757885
* #### Support Vector Machines (grid) = 0.998638529612
   *Best estimator:
(C=1000, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',max_iter=-1,     probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)*
* #### Decision Tree Classifier = 0.871341048332
* #### Random Forest Classifier = 0.823235761289
