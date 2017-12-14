import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wizualizacja danych
# Nazwa pliku z którego pobierane są dane
name = 'capture20110810.csv'
# Wczytanie pliku
df = pd.read_csv(name)
# Obliczenia korelacji pomiędzy kolumnami
corr = df.corr()
# Wielkość wykresu
plt.subplots(figsize=(15,10))
# Tworzenie wykresu typu Heatmap
sns.heatmap(corr, yticklabels=corr.columns.values, xticklabels=corr.columns.values)
# Określenie położenia opisów na osiach
plt.yticks(rotation=0)
plt.xticks(rotation=90)
# Zapisanie wykresu
plt.savefig('Heatmap_{}png'.format(name[:-3]),bbox_inches='tight', dpi=300)
# Zdefiniowanie dla których kolumn ma być wyrysowany "pairplot"
v = ['avg_duration', 'n_dports>1024', 'n_dports<1024', 'n_tcp', 'n_icmp', 'n_udp', 'n_d_a_p_address', 'n_d_b_p_address', 'n_d_c_p_address', 'n_d_na_p_address', 'n_s_a_p_address', 'n_s_b_p_address', 'n_s_c_p_address', 'n_s_na_p_address', 'n_sports>1024', 'n_sports<1024',  'background_flow_count', 'normal_flow_count', 'n_conn']
# Zdefiniowanie wykresu "pairplot"
sns.pairplot(df, vars=v, hue='label', palette='husl',size = 2.5)
# Zapis wykresu
plt.savefig('Pairplot_{}png'.format(name[:-3]),bbox_inches='tight', dpi=300)
