# Add import path for the dezero directory.

from dezero.core_simple import Variable
from dezero.core_simple import Function
from dezero.core_simple import using_config
from dezero.core_simple import no_grad
from dezero.core_simple import as_array
from dezero.core_simple import as_variable
from dezero.core_simple import setup_variable 

setup_variable() #연산자 오버로드가 이루어진 상태에서 Variable을 사용할수 있도록 



if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Python로 실행할 때, 상위 폴더를 path에 추가해서 모듈을 임포트할 수 있도록
    

import numpy as np
from dezero import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)

