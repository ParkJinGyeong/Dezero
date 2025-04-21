if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y
# 비선형 함수이며, 여러 개의 극값(극소/극대점)이 존재함

def gx2(x):
    # 2차 도함수 직접 계산한 부분 
    # 뉴턴법에서는 1차 도함수(기울기)와 함께 2차 도함수를 사용해서 이동 방향을 조절함
    return 12 * x ** 2 - 4

# Variable 타입으로 선언했기 때문에 자동 미분 가능
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
    # 함수의 국소 최소값에 빠르게 수렴

# Variable 기반 자동 미분 시스템은 1차 도함수를 자동 계산하고
# 2차 도함수는 직접 계산해야 함
# 뉴턴법 기반의 고차 최적화도 구현 가능