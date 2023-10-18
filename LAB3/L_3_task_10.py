import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import affinity_propagation
import yfinance as yf  # Використовуємо yfinance для отримання фінансових даних

# Ваш шлях до файлу з прив'язками символічних позначень компаній
with open('company_symbol_mapping.json', 'r') as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

# ...
# Завантаження архівних даних котирувань за допомогою yfinance
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2021, 12, 31)

quotes = []

# Отримання котирувань для всіх компаній
for symbol in symbols:
    try:
        quote = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
        if not quote.empty:
            quotes.append(quote)
        else:
            print(f"No data available for [{symbol}]")
    except Exception as e:
        print(f"Failed download: [{symbol}]: {e}")

# ...
# Обчислення середнього значення котирувань для всіх компаній
valid_quotes = [quote.values for quote in quotes if not quote.empty]
min_length = min(len(quote) for quote in valid_quotes)
closing_quotes = np.array([quote[:min_length] for quote in valid_quotes], dtype=float)

# Обчислення різниці між відкриттям та закриттям
quotes_diff = closing_quotes - closing_quotes[:, 0][:, None]

X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# ...
for i in range(num_labels + 1):
    cluster_names = names[np.where(labels == i)]
    print("Cluster", i + 1, "==>", ', '.join(cluster_names))



