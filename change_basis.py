import numpy as np
π = np.pi
e1, e2, e3 = np.eye(3)

def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    a3 = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b3 = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d3 = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return np.r_[a3, b3, c3, d3]

quat = lambda θ, u: np.r_[np.cos(θ/2), np.sin(θ/2)*u/np.linalg.norm(u)]
conj = lambda q: np.r_[q[0], -q[1:]]
rotv = lambda q, v: qmul(qmul(q, np.r_[0, v]), conj(q))[1:]
rotq = lambda q, p: np.r_[p[0], rotv(q, p[1:])]
rmat = lambda q: np.c_[rotv(q, e1), rotv(q, e2), rotv(q, e3)]

print("=== Change of vector coordinates under change of basis ===")
ξ_prime = np.array([2, 1, 3])
t = quat(π/5, np.array([2, -3, 1]))
T = rmat(t)
print(f"Matr: ξ = T @ ξ' = {T @ ξ_prime}")
print(f"Quat: ξ = t @ ξ' @ t^(-1) = {rotv(t, ξ_prime)}")

print("\n=== Change of rotation coordinates under change of basis ===")
q_prime = quat(π/4, np.array([1, 1, 0]))
print(f"Matr: Q = T @ Q' @ T^(-1) =\n{T @ rmat(q_prime) @ T.T}")
print(f"Quat: q = t @ q' @ t^(-1) => Matr\n{rmat(rotq(t, q_prime))}")