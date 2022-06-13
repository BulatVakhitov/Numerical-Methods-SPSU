import numpy as np


A = np.array([[4, 1, 1],
              [1, 6.6, -1],
              [1, -1, 8.6]])

b = np.array([1, -2, 3])


def f(x):
    return 0.5 * np.dot(np.dot(x.T, A), x) + np.dot(x.T, b) + 3


x0 = np.array([1, 0, 0])
x_current = x0
x_previous = np.array([0, 0, 0])
eps = 1e-6

e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

orts = [e1, e2, e3]
ort_index = 0

while abs(f(x_current) - f(x_previous)) > eps:
    current_ort = orts[ort_index]

    x_previous = x_current
    temp = np.dot(A, x_previous) + b
    mu = (-1) * np.dot(current_ort.T, temp) / np.dot(current_ort.T, np.dot(A, current_ort))

    x_current = x_previous + mu * current_ort
    ort_index = (ort_index + 1) % 3


print('\nf_min = ', f(x_current), '\nx_min = ', x_current,
      '\ndelta =', abs(f(x_current) - f(x_previous)))
