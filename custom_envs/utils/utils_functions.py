'''Module that holds different functions for use in problems.'''


def compute_rosenbrock(x, y):
    '''Compute Rosenbrock's banana function: f(x,y)=(1-x)^2+100(y-x^2)^2'''
    return 100*(y - x**2)**2 + (1 - x)**2
