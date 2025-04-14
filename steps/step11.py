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
            self.grad = np.ones_like(self.data)

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


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]  # 인수와 반환값을 리스트로 변환 # list comprehension을 사용함 
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]  

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs #xs 리스트에서 원소 두개 뽑아오기 
        y = x0 + x1 
        return (y,) #튜플 형식으로 반환하기 

 
xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)