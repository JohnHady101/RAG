def f(x):
    return x * 2

def g(y):
    return f(y)

print(g(3))  # This will print 6 because g returns the result of f(y)
