import numpy as np
import math
point1 = (1, 200)
point2 = (2, 150)
point3 = (3, 351)
x = tuple(zip(point1,point2,point3))

squares = [(p-q) for p,q in zip(point1,point2)]

print(squares)
for p,q,y in x:
    
    m = p+q-y
    print(m)
   

print(x)