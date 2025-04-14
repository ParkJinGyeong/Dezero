class Variable:
    def __init__(self, data):
        self.data = data 
import numpy as np
data = np.array(1.0)
x = Variable(data) 
print(x.data)

# 스칼라, 백터, 행렬 개념 언급
# numpy에서 n.dim 개념 언급 
# 백터를 다룰 때는 차원이라는 개념을 주의해야하는데 np.array([1,2,3])은 백터인데, 세개의 요소가 일렬로 있어서 3차원 백터라고 함. 
# 여기서 백터의 차원이란 원소의 수를 말하는거임. 한편 3차원 배열에서는 원소가 아니라 축이 3개라는 뜻  

class Function:
    def __call__(self,input): #call 매서드의 인수는 Variable의 인스턴스라고 가정 
        x = input.data #데이터를 꺼냄 
        y = x ** 2 # 실제 계산 
        output = Variable(y) #Variable 형태로 되돌린다 
        return output 
x = Variable(np.array(10))
f = Function()
y = f(x)

print(y.data)
print(type(y)) #y의 class는 Variable임 

class Function:
    def __call__(self,input): #데이터 찾기, 계산결과를 Variable로 포장하기 
        x = input.data 
        y = self.forward(x) 
        output = Variable(y)
        return output 
    def forward(self, x):
        raise NotImplementedError()
class Square(Function): #function 클래스를 상속 받아서 입력값을 제곱하는 클래스 
    def forward(self, x):
        return x**2
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    
x = Variable(np.array(10))
f = Square()
y = f(x)
print(y.data)
print(type(y))

#Function 클래스의 __call__ 메서드는 입력과 출력이 모두 Variable 인스턴스라서 자연스럽게 DeZero 함수들을 연이어 사용할수 있다
#입출력이 Variable 인스턴스로 통일 되어있어서 여러 함수를 연속하여 적용할수 있음- 미분을 효율적으로 계산할수 있는 배경
#전진차분보다 중앙차분이 진정한 미분값에 가깝다는 사실은 테일러급수를 이용해서 증명할수 있음 , 그런데 수치 미분은 계산량이 너무 많아서 역전파 개념이 등장하게 됨 
#역전파가 잘 되는지 확인하려면 기울기 확인을 해야함 
#전파되는 데이터는 모두 y의 미분값이라는 것. (오른쪽에서 왼쪽으로) 
#머신러닝은 주로 대량의 매개변수를 입력받아서 마지막에 손실함수를 거쳐 출력을 내는 형태로 진행되는데, 손실함수의 각 매개변수에 대한 미분을 계산해야함.
#미분값을 출력에서 입력방향으로 전파하면 한번의 전파만ㅇ으로 모든 매개변수에 대한 미분을 계산할수 있음


