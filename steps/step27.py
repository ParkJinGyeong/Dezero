if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    #사용자 정의 함수로 sin 구현 
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
         # dy/dx = cos(x)
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print('--- original sin ---')
print(y.data)
print(x.grad)


def my_sin(x, threshold=0.0001):
    #my_sin은 Variable 연산으로 구성되어 있어 역전파도 자동으로 가능
    y = 0
    for i in range(100000):
        # # 테일러 급수 항: (-1)^i / (2i+1)! * x^(2i+1)
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)  # ← 여기서 x는 Variable이니까 계산 그래프 생성됨
        y = y + t  # ← Variable + Variable → 계산 그래프가 계속 이어짐
        if abs(t.data) < threshold:
            break
    return y


x = Variable(np.array(np.pi / 4))
y = my_sin(x)  # , threshold=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin.png')