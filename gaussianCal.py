import math
mean = float(input('Mean:'))
var = float(input('Var:'))
x = float(input('X:'))

gb = float((1 / math.sqrt(2 * math.pi * var)) * math.exp((x * mean)**2 / (2 * var)))
print(gb)
