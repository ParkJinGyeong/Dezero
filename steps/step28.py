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


def rosenbrock(x0, x1): #Variable 객체
    
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y # 반환값까지 ! 

# 최적화 대상 변수 초기화
# 로젠브록 함수의 최소점 (1, 1)과 거리가 있는 위치에서 시작
x0 = Variable(np.array(0.0)) # 시작 위치 x=0
x1 = Variable(np.array(2.0)) # 시작 위치 y=2
lr = 0.001 
iters = 1000

for i in range(iters): # 매 반복마다 x0, x1의 값이 갱신되는 걸 추적


    print(x0, x1)

    y = rosenbrock(x0, x1)
# 순전파: 로젠브록 함수 계산
    x0.cleargrad()  # 이전의 gradient 초기화
    x1.cleargrad()
    y.backward()
 # 역전파 수행 → x0.grad, x1.grad가 계산됨
    x0.data -= lr * x0.grad # Gradient Descent: 파라미터 업데이트
    x1.data -= lr * x1.grad