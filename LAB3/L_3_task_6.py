import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Розділимо дані на навчальний та перевірочний набори
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Функція для побудови кривих навчання
def plot_learning_curves(model, X, y, X_val, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X)):
        model.fit(X[:m], y[:m])
        y_train_predict = model.predict(X[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), label="Навчальна помилка")
    plt.plot(np.sqrt(val_errors), label="Перевірочна помилка")
    plt.legend()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
X_poly_val = poly_features.transform(X_val.reshape(-1, 1))

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X_poly, y_train, X_poly_val, y_val)

plt.xlabel('Розмір навчального набору')
plt.ylabel('RMSE')
plt.title('Криві навчання для поліноміальної моделі (2-го ступеня)')
plt.show()
