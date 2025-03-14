import sympy as sp

# 定义方程 f(x) = 2x^3(4 - 3x) - 1
x = sp.symbols('x')
f = 2 * x**3 * (4 - 3 * x) - 1
f_prime = sp.diff(f, x)  # 计算 f'(x)

# 定义 Newton 方法
def newton_method(func, func_prime, x0, tolerance=1e-3, max_iter=100):
    """
    使用 Newton 方法求解方程 f(x) = 0 的近似解，并输出迭代过程。
    :param func: 原函数 f(x)
    :param func_prime: 原函数的导数 f'(x)
    :param x0: 初始猜测值
    :param tolerance: 误差容限
    :param max_iter: 最大迭代次数
    :return: 近似解
    """
    xn = x0
    print(f"Initial guess: x0 = {xn}")
    for i in range(max_iter):
        f_xn = func.evalf(subs={x: xn})
        f_prime_xn = func_prime.evalf(subs={x: xn})
        if abs(f_prime_xn) < 1e-8:  # 避免分母接近零
            print("Derivative too small, stopping iteration.")
            break
        xn_next = xn - f_xn / f_prime_xn
        print(f"Iteration {i + 1}: x = {xn_next}, f(x) = {f_xn}")
        if abs(func.evalf(subs={x: xn_next})) < tolerance:
            print("Converged!")
            return xn_next
        xn = xn_next
    print("Did not converge within the maximum iterations.")
    return xn

# 初始猜测值选择为 x0 = 0.5
initial_guess = 0.5
median = newton_method(f, f_prime, initial_guess)

# 打印最终结果
print(f"\nEstimated median: {median}")
print(f"Verification: f(median) = {f.evalf(subs={x: median})}")
