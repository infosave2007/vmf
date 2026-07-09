import sympy as sp
from sympy.polys.polyfuncs import symmetrize

# We want to evaluate symmetric polynomials in z1, z2, z3
# given e1 = 1, e2 = 0, e3 = -eps**2

eps = sp.Symbol('eps')
z1, z2, z3 = sp.symbols('z1 z2 z3')

e1 = 1
e2 = 0
e3 = -eps**2

def eval_sym(expr):
    # Symmetrize the expression in terms of elementary symmetric polynomials
    sym_expr, rem = symmetrize(expr, formal=True, polys=True)
    if rem != 0:
        return "Not symmetric!"
    
    # sym_expr is a polynomial in s1, s2, s3 (the elementary symmetric polynomials)
    # The output format is a bit tricky, let's just do substitution manually
    
    # We can use Newton's sums directly instead of symmetrize for simple powers
    pass

# Let's use a simpler algebraic approach for Newton's sums
# p_k = z1^k + z2^k + z3^k
p = {0: 3, 1: e1}
p[2] = e1*p[1] - 2*e2
p[3] = e1*p[2] - e2*p[1] + 3*e3
p[4] = e1*p[3] - e2*p[2] + e3*p[1]
p[5] = e1*p[4] - e2*p[3] + e3*p[2]

print("Newton Sums (p_k = sum z_i^k):")
for k in range(1, 6):
    print(f"p_{k} = {sp.expand(p[k])}")

# Let's check negative powers
# p_{-1} = sum 1/z_i = e2 / e3
p_minus1 = e2 / e3
# p_{-2} = sum 1/z_i^2 = (e2^2 - 2 e1 e3) / e3^2
p_minus2 = (e2**2 - 2*e1*e3) / e3**2

print("\nNegative Powers:")
print(f"p_-1 = {p_minus1}")
print(f"p_-2 = {sp.expand(p_minus2)}")

# Now let's check the thermodynamic quantities
# kappa_i is proportional to 3 - 2/z_i
print("\nThermodynamic Invariants:")
sum_kappa = 3*3 - 2*p_minus1
print(f"Sum of kappa = {sp.expand(sum_kappa)}")

# Area is proportional to z_i^2
# Work: sum A_i kappa_i = sum z_i^2 (3 - 2/z_i) = 3 p_2 - 2 p_1
sum_work = 3*p[2] - 2*p[1]
print(f"Sum of A*kappa = {sp.expand(sum_work)}")

# sum kappa_i A_i^(1/2) = sum z_i (3 - 2/z_i) = 3 p_1 - 2(3) = 3(1) - 6 = -3
sum_first_law = 3*p[1] - 2*3
print(f"Sum of kappa * sqrt(A) = {sp.expand(sum_first_law)}")

# What about sum kappa_i * A_i^2 ? (related to volume thermodynamics)
# sum z_i^4 (3 - 2/z_i) = 3 p_4 - 2 p_3
sum_kappa_A2 = 3*p[4] - 2*p[3]
print(f"Sum of kappa * A^2 = {sp.expand(sum_kappa_A2)}")

