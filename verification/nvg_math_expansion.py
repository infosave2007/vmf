import sympy as sp

eps = sp.Symbol('eps')

# Inner horizon: x = eps / sqrt(1-x)
# We can find the series for x up to O(eps^20)
x = 0
for i in range(25):
    x = eps * (1 - x)**(-sp.Rational(1, 2))
    x = x.series(eps, 0, 25).removeO()

print("Inner horizon root x(eps):")
print(x)
print()

# Outer horizon: y = eps^2 / (1-y)^2, where rout = rg(1-y)
delta = eps**2
y = 0
for i in range(15):
    y = delta * (1 - y)**(-2)
    y = y.series(eps, 0, 25).removeO()

print("Outer horizon deformation y(eps):")
print(y)
print()

# Entropy deficit
# Delta S / S_inf = (2y - y^2 - x^2) / eps^2
entropy_deficit = (2*y - y**2 - x**2) / eps**2
entropy_deficit = entropy_deficit.series(eps, 0, 20).removeO()
print("Delta S(M) / S_in(infty):")
print(sp.expand(entropy_deficit))

