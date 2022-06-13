import numpy as np

N = 3

A = np.array([[4, 1, 1],
              [1, 6.6, -1],  # коэффициент перед игреком = (3 + 0.1 * 3) = 3.3, а в матрице он удвоен
              [1, -1, 8.6]])

b = np.array([1, -2, 3])


def f(x):
    return 0.5 * np.dot(np.dot(x.T, A), x) + np.dot(x.T, b) + N


x0 = np.array([1, 0, 0])
x_current = x0
x_previous = np.array([0, 0, 0])
eps = 1e-6

while abs(f(x_current) - f(x_previous)) > eps:  # критерий остановки: абсолютная разность значений функции
    x_previous = x_current
    q = np.dot(x_previous, A) + b  # q = Ax + b
    mu = (-1) * np.dot(q.T, q) / np.dot(q.T, np.dot(A, q))  # находим мю, mu = - (q.T * q) / (q.T * A * q)

    x_current = x_previous + mu * q


print('\nf_min = ', f(x_current), '\nx_min = ', x_current,
      '\ndelta =', abs(f(x_current) - f(x_previous)))
