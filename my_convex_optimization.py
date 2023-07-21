import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

f = lambda x : (x - 1)**4 + x**2
res = minimize_scalar(f, method='brent')   
print('x_min: .02f, f(x_min): .02f', res.x, res.fun)
    
# plot curve
x = np.linspace(res.x - 1, res.x + 1, 100)
y = [f(val) for val in x]
plt.plot(x, y, color='blue', label='f')
    
# plot optima\\n",
plt.scatter(res.x, res.fun, color='red', marker='x', label='Minimum')
plt.grid(True)
plt.legend(loc = 1)

def print_a_function(f, values):
    plt.plot(f(values))
    return(f(values))


def find_root_bisection(f, min, max):
    x = min
    
    while ((max-min) >= 0.01):
        x = (min+max)/2
        if (f(x) == 0.0): 
            break   
        if (f(x)*f(min) < 0):  
            max = x   
        else:   
            min = x  
    return x  


find_root_bisection(f, -2, 3)


def find_root_newton_raphson(f, f_deriv, start, precision=0.001):
    x = start
    while abs(f(x)) > precision:
        x = x - f(x) / f_deriv(x)
    return x


# Define the derivative of f(x)
f_prime = lambda x: 4 * ((x - 1) ** 3) + 2 * x
def gradient_descent(f, f_prime, start, learning_rate = 0.1, precision=0.001):
    x = start
    while abs(f_prime(x)) > precision:
        x -= learning_rate * f_prime(x)
    return x

start = -1
x_min = gradient_descent(f, f_prime, start, 0.01)
f_min = f(x_min)

print("xmin: {:.2f}, f(x_min): {:.2f}".format(x_min, f_min))

# Plot the function with the gradient descent minimum
x = np.linspace(x_min - 1, x_min + 1, 100)
y = f(x)
plt.plot(x, y, color='blue', label= 'f')
plt.scatter(x_min, f_min, color='red', marker='x', label='Minimum (Gradient Descent)')
plt.title('Plot of the function f(x) with Gradient Descent Minimum')
plt.grid(True)
plt.legend()
plt.show()


def solve_linear_problem(A, b, c):
    from scipy.optimize import linprog

    res = linprog(c, A_ub=A, b_ub=b, method='simplex')

    return res.fun, res.x

# Linear problem with Simplex method
A = np.array([[2, 1], [-4, 5], [1, -2]])
b = np.array([10, 8, 3])
c = np.array([-1, -2])

optimal_value, optimal_arg = solve_linear_problem(A, b, c)

print("The optimal value is:", optimal_value, "and is reached for x =", optimal_arg)
