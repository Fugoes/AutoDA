import numpy as np

ADD = lambda a, b: a + b
SUB = lambda a, b: a - b
MUL = lambda a, b: a * b
DIV = lambda a, b: a / b
DOT = lambda a, b: np.dot(a, b)
NORM = lambda a: np.linalg.norm(a)

x = np.zeros(shape=32 * 32 * 3)
x_adv = np.zeros(shape=32 * 32 * 3)
noise = np.random.normal(size=32 * 32 * 3)


def test():
    s0 = 0.01
    v1 = x
    v2 = x_adv
    v3 = noise
    v4 = SUB(v1, v2)
    s5 = NORM(v4)
    v6 = DIV(v4, s5)
    v7 = MUL(v3, s0)
    v8 = ADD(v7, v6)
    s9 = NORM(v2)
    v10 = MUL(v8, s5)
    s11 = DOT(v10, v7)
    v12 = DIV(v6, s9)
    v17 = ADD(v6, v12)
    v20 = MUL(v17, s11)
    v21 = ADD(v1, v20)
    v23 = SUB(v21, v10)
    return v23
