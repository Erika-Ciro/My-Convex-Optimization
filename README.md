# My-Convex-Optimization

# Description
**Subject**

One of the main challenges of Data Science and more specifically in Machine Learning is the performance measure.
How to measure performance efficiently so that our model predictions meet the business objectives?

For example, let's pretend our goal is to classify fruits to know if a fruit is an apple, an orange, or something else, depending on some attributes (color, shape, weight...).

Intuitively, the measure of 'how far' is our model prediction from the correct answer tells us about the effectiveness of our algorithm. The farther is our prediction the worse our performance. We can then define a cost function based on this 'distance'. We want to minimize this ***cost function*** to have the best prediction possible, i.e. we want to find the point where the value of this cost function is the lowest.

Generally, minimizing a function is not an easy task. However, some functions are easier to optimize than others: Convex functions.

![image](https://github.com/Erika-Ciro/My-Convex-Optimization/assets/104148159/de99426b-bf29-4034-987f-9127dcd93154)


On the left, the function is said to be convex: it is beautifully round. On the contrary, the function on the right has a lot of valleys and bumps.
The minimum of the convex function is much easier to reach: we just need to follow the slope! With a non-convex function, following the slope would not work since reaching a cavity does not ensure that this cavity is the lowest one!

Finding the minimum of a convex function is called ... Convex Optimization! You might have already heard about something called Gradient Descent before. This is an algorithm that does exactly that: following the slope to the cavity.

This project is meant to be an introduction to some convex optimization tools and to implement your own optimization algorithm.

# Introduction

To get a feel with the problem, we are going to start with a simple function to optimize:

***f(x) = (x - 1)^4 + x^2***

Which translates in python to: ***f = lambda x : (x - 1)**4 + x**2***

→ ***Plot this function to get a feel of what it looks like!***

To find the minimum of our little f function, we are going to use its derivative f' (f prime). The zero of f prime (the point where it evaluates to zero) matches with the minimum of the f function.

Our goal is hence to find where f prime cancels out.

→ ***Write a simple dichotomous algorithm (bisection method) to find the zero of a function.***

**def find_root(f, a, b)** should return x such that **f(x) = 0.** Since computers only understand discrete values, we will have a precision of **0.001**, i.e. find x such that **|f(x)| < 0.001 (|.| is the absolute value function).**

If your dichotomous find_root function works well, you can try other root-finding algorithms like **Newton-Raphson's or Muller's method** (but they are many more).

Once you are done playing around with **root-finding methods**, we will use find_root to find the minimum of f. As explained above, the root of f', the derivative of f, is where f reaches its minimum.

→ ***Use find_root to find the root of f prime.***

***f' = 4*(x - 1)^3 + 2x**

To make sure the answer is correct, we will check it against **Brent's method** for optimization. Brent's method works as a combination of the secant method and parabola fittings.



# Gradient Descent Methods

Suppose you are lost at the top of a hill at night. You cannot see anything, you can only feel the ground beneath your feet. An intuitive way to reach the village downhill as fast as possible is to follow the steepest slope.

This is the idea behind the gradient descent. It measures the local gradient of the cost function and goes toward the direction of the descending gradient. Once the gradient cancels out, it means we reached a minimum.

**xk + 1 = xk - α ∇f(xk)**

where **∇f** is the gradient of f and α is called the **learning rate**. This is how big you step towards the direction of the descending gradient.

In our one dimension example, the gradient of f is simply the derivative of f.



→ **How does the learning rate influence the efficiency of the algorithm? What happens if it is very small? What if it is very big?**

→ ***Write a simple gradient descent function that finds the minimum of a function f***


Is the result similar to previous Brent's method?

Using the gradient_descent, you should find a value similar to the previous methods.

Gradient Descent methods are the workhorse of machine learning, from linear regression to deep neural nets.
Here, we used it in a one-dimension problem, but it can be used with any number of dimensions!

Similarly, we will stop our search with a precision of 0.001.


# To go further

Adding linear constraints to a convex function does not change its convexity, it remains convex. What it does though is restricting the space of solutions by intersecting it with hyperplanes.
If our convex function solution space is a spherical orange, applying linear constraints is like slicing the orange. It remains convex but sharper.

More specifically, let us consider the following linear problem with two variables:

**maximize   z = x + 2y
subject to
      • 2x + y ≤ 10
      • -4x + 5y ≤ 8
      • x - 2y ≤ 3
      • x, y ≥ 0**

Which can be rewritten as:

**maximize   z = cT·x
subject to
      • Ax ≤ b
      • x ≥ 0**

z is called the objective function, A a coefficients matrix, and b the non-negative constraints. The space defined by the constraints equations is called the feasible region.
This is a convex polytope. It can be shown that solutions that maximize the objective function are located on the vertices of this polytope.

# Simplex algorithm

We are going to solve this linear problem with the Simplex method.
The simplex algorithm is pretty straightforward: it moves from one vertex to another until it finds a solution that maximizes the objective function.

→ ***Solve the linear problem using the simplex method***

**hint: You can use scipy's implementation of Simplex.**

→ ***Is the solution you found located on the edge of the polytope? Why?***

