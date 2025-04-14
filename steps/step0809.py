class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data  # 데이터 저장
        self.grad = None  # 미분 저장
        self.creator = None
    def backward(self):
        f= self.creator 
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
            # 재귀문 
    def backward(self):  # 역전파
 
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수 가져옴 
            x, y = f.input, f.output # 함수의 입력과 출력을 가져옴 
            x.grad = f.backward(y.grad) # backward 매서드 호출 ㄷ

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트를 추가함 
        
        #반복문이 훨씬 효율적임 


import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #self.data와 형상과 데이터 타입이 같은 ndarray 인스턴스를 생성하는데 모든 요소를 1로 채워서 돌려줌 .
            #예를들어서 self.data가 스칼라면 self.grad도 스칼라 형식으로 

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
# 스칼라 타입인지 확인해줌 


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) #항상 np.array 형식으로 사용할수 있음
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)


x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
x = Variable(1.0)  # NG