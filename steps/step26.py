import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph 

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

# 연산 수행 (자동으로 Add Function이 연결됨)
y = x0 + x1 

#변수 이름 지정 
x0.name = "x0"
x1.name = "x1"
y.name = "y"

# 연산 그래프를 DOT 언어 형식 문자열로 출력
txt = get_dot_graph(y, verbose=False) # verbose: 노드에 데이터 타입/shape까지 표시할지 여부

#dot 파일로 저장 
with open('simple.dot', 'w') as o:
    o.write(txt)

  # 변수 노드를 DOT 언어로 생성  
def _dot_var(v, verbose = False) :
    dot_var = '{} [label = "{}", color = orange, style = filled]\n'
    
    name = '' if v.name is None else v.name  # 이름이 있으면 이름 사용
    if verbose and v.data is not None: 
        if v.name is not None: 
            name += ':'
        name += str(v.shape) + '' + str(v.dtype)  # verbose이면 shape, dtype도 같이 출력
    return dot_var.format(id(v), name) # Variable 객체의 id를 노드 ID로 사용

def _dot_func(f):
    dot_func = '{} [label = "{}", color = lightblue, style = filled, shape = box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    
    dot.edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # 함수 → 출력 (outputs는 약한 참조라 y()로 꺼냄)
    return txt 
# 전체 계산 그래프를 순회하며 DOT 문자열 생성 / 전체 흐름 
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []         # 순회할 Function 저장용 스택
    seen_set = set()   # 중복 방문 방지

    # Function이 중복되지 않게 등록
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    # 출력 변수의 creator부터 시작
    add_func(output.creator)

    # 출력 변수도 그래프에 추가
    txt += _dot_var(output, verbose) 

    # 그래프 순회 (역전파처럼 거꾸로 따라감)
    while funcs:
        func = funcs.pop() #function 하나씩 꺼내기 
        txt += _dot_func(func)  #function 노드 출력

        for x in func.inputs:
            txt += _dot_var(x, verbose)  # 입력 변수 노드 출력

            if x.creator is not None:
                add_func(x.creator)  # 입력 변수가 또 다른 함수의 출력이면 추가로 따라감

    return 'digraph g {\n' + txt + '}'  # DOT 언어 전체 그래프 구조로 감싸서 반환