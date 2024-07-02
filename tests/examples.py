import numpy as np

def lp_func(point, compute_hessian=False):
    objective = -point[0] - point[1]
    gradient = np.array([-1, -1])

    if compute_hessian:
        hessian = np.zeros((2, 2))
    else:
        hessian = None

    return objective, gradient.T, hessian

def lp_ineq_1(point, compute_hessian=False):
    constraint = -point[0] - point[1] + 1
    gradient = np.array([-1, -1])

    if compute_hessian:
        hessian = np.zeros((2, 2))
    else:
        hessian = None

    return constraint, gradient.T, hessian

def lp_ineq_2(point, compute_hessian=False):
    constraint = point[1] - 1
    gradient = np.array([0, 1])

    if compute_hessian:
        hessian = np.zeros((2, 2))
    else:
        hessian = None

    return constraint, gradient.T, hessian

def lp_ineq_3(point, compute_hessian=False):
    constraint = point[0] - 2
    gradient = np.array([1, 0])

    if compute_hessian:
        hessian = np.zeros((2, 2))
    else:
        hessian = None

    return constraint, gradient.T, hessian

def lp_ineq_4(point, compute_hessian=False):
    constraint = -point[1]
    gradient = np.array([0, -1])

    if compute_hessian:
        hessian = np.zeros((2, 2))
    else:
        hessian = None

    return constraint, gradient.T, hessian

def qp_function(point, compute_hessian=False):
    objective = point[0]**2 + point[1]**2 + (point[2] + 1)**2
    gradient = np.array([2*point[0], 2*point[1], 2*(point[2] + 1)])

    if compute_hessian:
        hessian = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    else:
        hessian = None

    return objective, gradient.T, hessian

def qp_inq1(point, compute_hessian=False):
    constraint = -point[0]
    gradient = np.array([-1, 0, 0])

    if compute_hessian:
        gradient = gradient.T
        hessian = np.zeros((3, 3))
    else:
        hessian = None

    return constraint, gradient, hessian

def qp_inq2(point, compute_hessian=False):
    constraint = -point[1]
    gradient = np.array([0, -1, 0])

    if compute_hessian:
        gradient = gradient.T
        hessian = np.zeros((3, 3))
    else:
        hessian = None

    return constraint, gradient, hessian

def qp_inq3(point, compute_hessian=False):
    constraint = -point[2]
    gradient = np.array([0, 0, -1])

    if compute_hessian:
        gradient = gradient.T
        hessian = np.zeros((3, 3))
    else:
        hessian = None

    return constraint, gradient, hessian

def qp_equality(point):
    return point[0] + point[1] + point[2]

