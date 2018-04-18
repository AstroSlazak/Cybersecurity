import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

# Ścieżka dostępu do folderu z danymi
Path = r"C:/Users/KamilM/Documents/GitHub/Cybersecurity/CTU-13-Dataset/"

# Wczytanie danych
def read_data(data_path):
    df = pd.read_csv(data_path)
    # print(df.head())
    X = df[['avg_duration', 'n_dports>1024', 'n_dports<1024', 'n_tcp', 'n_icmp', 'n_udp', 'n_d_a_p_address', 'n_d_b_p_address', 'n_d_c_p_address', 'n_d_na_p_address', 'n_s_a_p_address', 'n_s_b_p_address', 'n_s_c_p_address', 'n_s_na_p_address', 'n_sports>1024', 'n_sports<1024',  'background_flow_count', 'normal_flow_count', 'n_conn']].values
    y = df['label'].values
    # Encode - zamiana Normal i Attack na 01 lub 10
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    # print(X.shape)
    return(X, Y)
# Zdefiniowanie funkcji encoder'a
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Zdefiniowanie modelu
def multilayer_perceptron(x, weights, biases):

    # Warstwa 1 z aktywacją sigmoid
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Warstwa 2 z aktywacją sigmoid , wejście to wyjście warstwy 1 x----> layer_1 , z nowymi wagami i bias'em
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Warstwa 3 z aktywacją sigmoid , wejście to wyjście warstwy 2 x----> layer_2 , z nowymi wagami i bias'em
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Warstwa 4 z aktywacją sigmoid , wejście to wyjście warstwy 3 x----> layer_3 , z nowymi wagami i bias'em
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    # Warstwa 5 z aktywacją sigmoid
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.sigmoid(layer_5)

    # Warstwa 6 z aktywacją sigmoid
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.sigmoid(layer_6)

    # Warstwa 7 z aktywacją sigmoid
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.sigmoid(layer_7)

    # Warstwa 8 z aktywacją sigmoid
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.sigmoid(layer_8)

    # warstwa wyjściowa z nowymi wagami i bias'em
    out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
    return out_layer

# Utworzenie notatnika do zazpisania skuteczności
with open("TF_accuracy.txt", "w") as output:
    # Przeiterowanie przez wszystkie foldery z danymi
    for n in range(1,14):
        filelist = os.listdir(Path+str(n))
        for i in filelist:
            if i.endswith(".csv"):
                data_path = Path+str(n)+"/"+i
                print(data_path)

                # Wywołanie funkcji
                X ,Y = read_data(data_path)
                # Tasowanie(mieszanie) bazy rzędami
                X, Y = shuffle(X, Y, random_state = 415)
                # Stworzenie danych testowych i ćwiczeniowych (30% bazy do testów)
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30 ,random_state = 101)
                # Sprawdzenie rozmiarów danych testowych i ćwiczeniowych
                # print(x_train.shape)
                # print(y_train.shape)
                # print(x_test.shape)
                tf.reset_default_graph()
                # zdefiniowanie liczby "ukrytych warstw" i ich liczby neuronów
                n_hidden_1 = 128
                n_hidden_2 = 256
                n_hidden_3 = 256
                n_hidden_4 = 512
                n_hidden_5 = 512
                n_hidden_6 = 256
                n_hidden_7 = 256
                n_hidden_8 = 128

                # Zdefiniowanie ważnych parametrów dla uczenia
                learning_rate = 0.0001 #Krok uczenia
                trainig_epochs = 500 #liczba iteracji
                cost_history = np.empty(shape=[1] ,dtype=float)
                n_dim = X.shape[1] #liczba wejść
                n_class = 2 #liczba wyjść

                # zdefiniowanie miejsca dla wejścia wyjścia bias'a i wagi
                x = tf.placeholder(tf.float32, [None, n_dim]) # Wejście0o  b
                W = tf.Variable(tf.zeros([n_dim, n_class])) # Wagi
                b = tf.Variable(tf.zeros([n_class])) # Bias
                y_ = tf.placeholder(tf.float32, [None, n_class]) # Wyjście

                # Zdefiniowanie wag i bias'a dla każdej warstwy (jako słownik)
                weights = {
                    'h1': tf.Variable(tf.random_normal([n_dim, n_hidden_1])),
                    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
                    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
                    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
                    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
                    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
                    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
                    'out': tf.Variable(tf.random_normal([n_hidden_8, n_class]))
                }
                biases = {
                    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
                    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
                    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
                    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
                    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
                    'b8': tf.Variable(tf.random_normal([n_hidden_8])),
                    'out': tf.Variable(tf.random_normal([n_class]))
                }

                # Wywołanie modelu
                y = multilayer_perceptron(x, weights, biases)

                # zdefiniowanie odchylenia(strat) i optymalizacji
                cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
                training_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)

                # Zainicjowanie wszystkich zmiennych
                init = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(init)

                # Obliczenia odchylenia(strat) i skuteczność dla każdej iteracji
                mse_history = []
                accuracy_history = []

                for epoch in range(trainig_epochs):
                    sess.run(training_step, feed_dict={x: x_train, y_: y_train})
                    cost = sess.run(cost_function, feed_dict={x: x_train, y_: y_train})
                    cost_history = np.append(cost_history, cost)
                    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    # print("Accuracy: ", (sess.run(accuracy, feed_dict={x: x_train, y_: y_train})))
                    pred_y = sess.run(y, feed_dict={x: x_test})
                    mse = tf.reduce_mean(tf.square(pred_y - y_test))
                    mse_ = sess.run(mse)
                    mse_history.append(mse_)
                    accuracy = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
                    accuracy_history.append(accuracy)

                    print('epoch: ', epoch, ' - ', 'cost: ', cost, "- MSE: ", mse_ , " - Train Accuracy: ", accuracy)
                # Zapis modelu sieci
                saver = tf.train.Saver()
                saver.save(sess, r'C:\Users\KamilM\Documents\GitHub\Cybersecurity\Model_TF\{}.ckpt'.format(i[:-4]))
                # zapisanie skuteczność do notatnika
                output.write("Skuteczność {} wynosi {} \n".format(n,accuracy_history[-1]))

                # Stworzenie wykresu skuteczności od iteracji
                plt.plot(accuracy_history,label="Przypadek nr {} ".format(n))
                plt.xlabel('Ilość iteracji')
                plt.ylabel('Skuteczność')
                plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.savefig('Model_tensorflow_{}.png'.format(learning_rate),bbox_inches='tight', dpi=300)
