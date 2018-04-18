# Cybersecurity
Projekt przejściowy - cybersecurity

[Baza danych](https://mcfp.weebly.com/the-ctu-13-dataset-a-labeled-dataset-with-botnet-normal-and-background-traffic.html)

#### Wykresy danych dla pierwszego scenariusza:
* Heatmap:
![](http://imagizer.imageshack.us/a/img924/7729/Da76Qq.png )
* Pairplot dla 3 wybranych kolumn:
![](http://imageshack.com/a/img924/6415/WQo4WO.png)
[Pairplot dla wszystkich kolumn](http://imageshack.com/a/img923/9964/zfNx4e.png)

#### Do określenia anomanlii użytezostały następijace kolumny:
 * n_tcp, n_udp, n_icmp, background_flow_count, normal_flow_count, n_conn
### Przy użyciu scikit-learn skuteczność wykrywania anomalii wynosi:

* #### Logistic Regression = 1.0
* #### Support Vector Machines = 0.846380757885
* #### Support Vector Machines (grid) = 0.998638529612
   *Best estimator:
(C=1000, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',max_iter=-1,     probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)*
* #### Decision Tree Classifier = 0.871341048332
* #### Random Forest Classifier = 0.823235761289
### Przy użyciu tensorflow skuteczność wykrywania anomalii wynosi:
* #### Tensorflow = 0.995689
#### Wykres zależności ilości iteracji od skuteczności:
![](https://ibb.co/dCzqMn)
