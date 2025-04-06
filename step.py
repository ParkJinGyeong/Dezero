import weakref
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(
                data, np.ndarray
            ):  # np.ndarray 값이 아니면 오류, ex) 리스트 등의 값이면 오류
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data  # 실제 데이터
        self.grad = None  # 미분 초기 값은 0
        self.creator = None
        self.generation = 0

    def set_creator(self, func):  # self는 func 통해 만들어진 Varicable
        self.creator = func
        self.generation = func.generation + 1  # 역전파 시 정렬 용

    def cleargrad(self):  # grad 초기화
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):  # Function들을 세대순으로 정렬하며 큐에 추가
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
            # 역전파 시 topological sort를 보장

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 역전파할 함수들을 하나씩 꺼냄 (뒤에서부터)
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


for i in range(10):
    x = Variable(np.random.randn(10000))  # big data
    y = square(square(square(x)))
