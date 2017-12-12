import pandas as pd
from functools import reduce
import ipaddress
import numpy as np

# Wczytanie danych
dataset_path = r"\...\..\.binetflow"
df = pd.read_csv(dataset_path)

# Rozpoczęcie  czyszczenia i przygotowywania danych do dalszej obróbki
# Usunięcie dziesiątych części sekundy
df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
# Zamiana argumentu na czas
df['StartTime'] = pd.to_datetime(df['StartTime'])
# ustawienie daty jako indeks
df = df.set_index('StartTime')
# Zamiana wartości NaN w Dport na port "-1"
df['Dport'] = df['Dport'].fillna('-1')
# Zamiana na numeru portu na integer int(x,0) <- zero oznacza ,że wartośćpozostaj zapisana w ten sam sposób
df['Dport'] = df['Dport'].apply(lambda x: int(x,0))
# To samo dla Sport
df['Sport'] = df['Sport'].fillna('-1')
df['Sport'] = df['Sport'].apply(lambda x: int(x,0))

"""
Dla interwału czasowego w tym przypadku 1s przepływu danych zostaną zsumowane i stworzona zostanie nowa baza z następującymi danymi:

    Rozszerzenia:
    0.  n_conn                       <-- Ilość ruchów w sieci                                  Pierwszy         Adres                            Liczba        Liczba hostów w
    1.  background_flow_count        <-- Typu ruchu                                    Klasa     oktet		    sieci	                         sieci	       jednej sieci
    2.  n_s_a_p_address              <-- Adres źródła sklasyfikowany jako "A"            A   |  1 – 126    |  0 N.H.H.H    |  255.0.0.0      |  126	       |  16,777,214 (224 – 2)
    3.  n_s_b_p_address              <-- Adres źródła sklasyfikowany jako "B"            B	 |  128 – 191  |  10 N.N.H.H   |  255.255.0.0	 |  16,382	   |  65,534 (216 – 2)
    4.  n_s_c_p_address              <-- Adres źródła sklasyfikowany jako "C"            C	 |  192 – 223  |  110 N.N.N.H  |  255.255.255.0	 |  2,097,150  |  254 (28 – 2)
    5.  n_s_na_p_address             <-- Adres źródła sklasyfikowany jako N/A
    6.  avg_duration                 <-- Średni czas trwania
    7.  n_d_a_p_address              <-- Adres celu sklasyfikowany jako "A"
    8.  n_d_b_p_address              <-- Adres celu sklasyfikowany jako "B"
    9.  n_d_c_p_address              <-- Adres celu sklasyfikowany jako "C"
    10. n_d_na_p_address             <-- Adres celu sklasyfikowany jako N/A
    11. n_dports > 1024              <-- Numer portu celu większy niż 1024
    12. n_dports < 1024              <-- Numer portu celu mniejszy niż 1024
    13. n_sports < 1024              <-- Numer portu źródła mniejszy niż 1024
    14. n_sports > 1024              <-- Numer portu źródła wiekszy niż 1024
    15. n_icmp                       <-- Typ protokołu icmp
    16. n_udp                        <-- Typ protokołu udp
    17. n_tcp                        <-- Typ protokołu tcp
    18. normal_flow_count            <-- Typu ruchu
    19. n_ipv6                       <-- Adres sklasyfikowany jako IPv6
"""
# print df.head()
# Zdefinowanie klas IP
def classify_ip(ip):
    try:
        ip_addr = ipaddress.ip_address(ip)
        # Sprawdzenie czy adres IP jest adresem IPv6
        if isinstance(ip_addr,ipaddress.IPv6Address):
            return 'ipv6'
        # Jeżeli nie to zostaje sprawdzony do której klasy należy
        elif isinstance(ip_addr,ipaddress.IPv4Address):
            # Rozbicie i Zamiana na kod szesnastkowy
            octs = ip_addr.exploded.split('.')
            # Sprawdzenie do której klasy należy
            if 0 < int(octs[0]) < 127: return 'A'
            elif 127 < int(octs[0]) < 192: return 'B'
            elif 191 < int(octs[0]) < 224: return 'C'
            # Jeżeli do żadnej zwraca N/A
            else: return 'N/A'
    # Dla wyjątków też zwraca N/A
    except ValueError:
        return 'N/A'
# Obliczenie średniego czasu trwania ruchu
def avg_duration(x):
    return np.average(x)
# Zwaraca sumę Dportów większych od 1024
def n_dports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
# nazwa zostaje przypisana jako 'n_dports>1024'
n_dports_gt1024.__name__ = 'n_dports>1024'
# Zwaraca sumę Dportów mniejszych od 1024
def n_dports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
n_dports_lt1024.__name__ = 'n_dports<1024'
# Zwaraca sumę Sportów większych od 1024
def n_sports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
n_sports_gt1024.__name__ = 'n_sports>1024'
# Zwaraca sumę Sportów mniejszych od 1024
def n_sports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
n_sports_lt1024.__name__ = 'n_sports<1024'
# Zwaraca oznaczenie dla bota 'Attack' dla normalnego ruchu 'Normal'
def label_atk_v_norm(x):
    for l in x:
        if 'Botnet' in l: return 'Attack'
    return 'Normal'
label_atk_v_norm.__name__ = 'label'
# Zwraca liczbę ruchów w tle
def background_flow_count(x):
    count = 0
    for l in x:
        if 'Background' in l: count += 1
    return count
# Zwraca liczbę ruchów normalnych w sieci
def normal_flow_count(x):
    if x.size == 0: return 0
    count = 0
    for l in x:
        if 'Normal' in l: count += 1
    return count
# Zwaraca całkowitą ilość ruchów w sieci
def n_conn(x):
    return x.size
# Zwaraca ilość użytego protokołu tcp
def n_tcp(x):
    count = 0
    for p in x:
        if p == 'tcp': count += 1
    return count
# Zwaraca ilość użytego protokołu udp
def n_udp(x):
    count = 0
    for p in x:
        if p == 'udp': count += 1
    return count
# Zwaraca ilość użytego protokołu icmp
def n_icmp(x):
    count = 0
    for p in x:
        if p == 'icmp': count += 1
    return count
# Zwaraca ilość IPv4 typu A źródeł
def n_s_a_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'A': count += 1
    return count
# Zwaraca ilość IPv4 typu A celu
def n_d_a_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'A': count += 1
    return count
# Zwaraca ilość IPv4 typu B źródeł
def n_s_b_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'B': count += 1
    return count
# Zwaraca ilość IPv4 typu B celu
def n_d_b_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'B': count += 1
    return count
# Zwaraca ilość IPv4 typu C źródeł
def n_s_c_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'C': count += 1
    return count
# Zwaraca ilość IPv4 typu C celu
def n_d_c_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'C': count += 1
    return count
# Zwaraca ilość niezidentyfikowanych IP zródeł
def n_s_na_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'N/A': count += 1
    return count
# Zwaraca ilość niezidentyfikowanych IP celu
def n_d_na_p_address(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'N/A': count += 1
    return count
# Zwaraca ilość IPv^
def n_ipv6(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'ipv6': count += 1
    return count


#Zdefiniowanie dla których kolumn mają być wykorzystane poszczególne funkcje tworząc nowe kolumny
extractors = {
    'Label'   : [label_atk_v_norm,
                 background_flow_count,
                 normal_flow_count,
                 n_conn,
                ],
    'Dport'   : [n_dports_gt1024,
                 n_dports_lt1024
                ],
    'Sport'   : [n_sports_gt1024,
                 n_sports_lt1024,
                ],
    'Dur'     : [avg_duration,
                ],
    'SrcAddr' : [n_s_a_p_address,
                 n_s_b_p_address,
                 n_s_c_p_address,
                 n_s_na_p_address,
                ],
    'DstAddr' : [n_d_a_p_address,
                 n_d_b_p_address,
                 n_d_c_p_address,
                 n_d_na_p_address,
                ],
    'Proto'   : [n_tcp,
                 n_icmp,
                 n_udp,
                ],
}
# Ustawienie bazy bedług sekundy
r = df.resample('1S')
# wywołanie funkcjii
n_df = r.agg(extractors)
# Pozbycie sie nadpisanej kolumny
n_df.columns = n_df.columns.droplevel(0) 
# print(n_df.head())
# zapisanie danych jako csv
n_df.to_csv(dataset_path[-25:-10] + '.csv')
