import numpy as np
from dezero import Variable
from dezero utils import get_dot_graph 

x0 = Variable(np.array(1.0))
x1 = Variable(np/array(1.0))
y = x0 + x1 

#변수 이름 지정 
x0.name = "x0"
x1.name = "x1"
y.name = "y"

txt = get_dot_graph(y, verbose=Fales)
print(txt)

#dot 파일로 저장 
with open('simple.dot', 'w') as o:
    o.write(txt)
    
def _dot_var(v, verbodse = False) :
    dot_var = '{} [label = "{}", color = orange, style = filled]\n'
    
    name = '' if v.name is None else v.name 
    if verbose and v.data is not None: 
        if v.name is not None: 
            name += ':'
        name += str(v.shape) + '' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_dunc(f):
    dot_func = '{} [label = "{}", color = lightblue, style = filled, shape = box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    
    dot.edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt 

def get_dot_graph(output, verbose = True):
    txt = ''
    funcs = []
    seen_set = set()
    
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot+var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'