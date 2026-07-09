from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def q_func_factory(x_min, y_min, dx, dy, q_array):
    nx, ny = q_array.shape

    def q_func(x, y):
        i = int((x - x_min) / dx)
        j = int((y - y_min) / dy)
        i = np.clip(i, 0, nx - 1)
        j = np.clip(j, 0, ny - 1)
        return q_array[i, j]

    return q_func


"""Utility functions for PDE solves and potentials used in the 2D notebook.

Functions provided:
- make_grid: create X,Y arrays and dx,dy
- default_D, default_V, default_gradV, default_laplacianD: default potential basis and helpers
- solve_committor_upwind: upwind committor solver (returns q on grid)
- weighted_diffusion_system: assemble weighted diffusion matrix (M-field)
- solve_u_w: convenience to compute M,g and solve for u and w
- estimate_D_on_boundary: average D on masked boundary points

The functions are written to accept user-supplied potential (V, gradV) and D, so
changing potential is as simple as defining new `V(x,y,theta)` and `gradV(x,y,theta)` functions
and passing them to the solvers.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Callable, Tuple

Array2 = np.ndarray


def make_grid(x_range: Tuple[float, float], y_range: Tuple[float, float], nx: int, ny: int):
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    return X, Y, dx, dy


def default_D(X: Array2, Y: Array2):
    """Default basis D(x) = [x^2-1, y^2] evaluated on grid X,Y.
    Returns array with shape (nx,ny,2).
    """
    return np.stack([(X ** 2 - 1.0), (Y ** 2)], axis=2)


def default_V(X: Array2, Y: Array2, theta: np.ndarray):
    Dvals = default_D(X, Y)
    # theta shape (2,) -> dot over last axis
    return Dvals @ theta


def default_gradV(X: Array2, Y: Array2, theta: np.ndarray):
    # returns (Vx, Vy)
    Vx = 2.0 * theta[0] * X
    Vy = 2.0 * theta[1] * Y
    return Vx, Vy


def default_laplacianD(x: float, y: float):
    # returns vector of second derivatives for components [x^2-1, y^2]
    return np.array([2.0, 2.0])


def solve_committor_upwind(nx: int, ny: int, x_range: Tuple[float, float], y_range: Tuple[float, float],
                           theta: np.ndarray, gradV_func: Callable[[Array2, Array2, np.ndarray], Tuple[Array2, Array2]],
                           beta: float = 1.0, radius: float = 0.1) -> Array2:
    """Solve committor on rectangular grid with upwind advection from gradV_func.

    gradV_func should accept X,Y,theta and return (Vx,Vy) arrays.
    """
    X, Y, dx, dy = make_grid(x_range, y_range, nx, ny)
    dist_A = np.sqrt((X + 1) ** 2 + Y ** 2)
    dist_B = np.sqrt((X - 1) ** 2 + Y ** 2)
    mask_A = dist_A <= radius
    mask_B = dist_B <= radius

    Vx, Vy = gradV_func(X, Y, theta)
    inv_beta = 1.0 / beta

    def k(i, j):
        return i * ny + j

    A = sp.lil_matrix((nx * ny, nx * ny))
    rhs = np.zeros(nx * ny)

    for i in range(nx):
        for j in range(ny):
            idx = k(i, j)

            if mask_A[i, j]:
                A[idx, idx] = 1.0
                rhs[idx] = 0.0
                continue
            if mask_B[i, j]:
                A[idx, idx] = 1.0
                rhs[idx] = 1.0
                continue

            # external Neumann
            if i == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i + 1, j)] = -1.0
                continue
            if i == nx - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i - 1, j)] = -1.0
                continue
            if j == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j + 1)] = -1.0
                continue
            if j == ny - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j - 1)] = -1.0
                continue

            c = -2.0 * inv_beta / dx ** 2 - 2.0 * inv_beta / dy ** 2
            cxm = inv_beta / dx ** 2
            cxp = inv_beta / dx ** 2
            cym = inv_beta / dy ** 2
            cyp = inv_beta / dy ** 2

            vx = Vx[i, j]
            vy = Vy[i, j]

            if vx >= 0.0:
                c += -vx / dx
                cxm += vx / dx
            else:
                c += vx / dx
                cxp += -vx / dx

            if vy >= 0.0:
                c += -vy / dy
                cym += vy / dy
            else:
                c += vy / dy
                cyp += -vy / dy

            A[idx, k(i, j)] = c
            A[idx, k(i - 1, j)] = cxm
            A[idx, k(i + 1, j)] = cxp
            A[idx, k(i, j - 1)] = cym
            A[idx, k(i, j + 1)] = cyp

    q_flat = spsolve(A.tocsr(), rhs)
    return q_flat.reshape((nx, ny))


def build_symbolic_helpers(D_exprs, theta_len=1, var_names=('x', 'y')):
    """Create numeric helpers (lambdified) from symbolic D expressions.

    D_exprs can be a list of sympy expressions or strings like ["x**2-1", "y**2"].
    theta_len sets how many theta symbols are created (theta0..).

    Returns a dict with callables:
      - D_func(X,Y) -> (..., nD)
      - V_func(X,Y, theta) -> (...)
      - gradV_func(X,Y, theta) -> (Vx, Vy)
      - lapV_func(X,Y, theta) -> (...)
      - gradientD_func(X,Y) -> (..., nD, 2) (dDi/dx, dDi/dy)
      - laplacianD_func(X,Y) -> (..., nD)
    """
    try:
        import sympy as sp
    except Exception as e:
        raise ImportError("sympy is required for build_symbolic_helpers") from e

    x_sym, y_sym = sp.symbols(var_names)
    theta_syms = sp.symbols([f"theta{i}" for i in range(theta_len)]) if theta_len > 0 else ()

    # parse D expressions
    D_syms = [sp.sympify(expr) for expr in D_exprs]

    # V = sum theta_i * D_i
    V_sym = sum((theta_syms[i] if i < len(theta_syms) else sp.Symbol(f'theta{i}')) * D_syms[i]
                for i in range(len(D_syms)))

    Vx_sym = sp.diff(V_sym, x_sym)
    Vy_sym = sp.diff(V_sym, y_sym)
    lapV_sym = sp.diff(V_sym, x_sym, 2) + sp.diff(V_sym, y_sym, 2)

    gradD_syms = [(sp.diff(Di, x_sym), sp.diff(Di, y_sym)) for Di in D_syms]
    lapD_syms = [sp.diff(Di, x_sym, 2) + sp.diff(Di, y_sym, 2) for Di in D_syms]

    # lambdify per-component for robustness
    D_funcs = [sp.lambdify((x_sym, y_sym), Di, modules='numpy') for Di in D_syms]
    V_func_l = sp.lambdify((x_sym, y_sym) + tuple(theta_syms), V_sym, modules='numpy')
    Vx_func_l = sp.lambdify((x_sym, y_sym) + tuple(theta_syms), Vx_sym, modules='numpy')
    Vy_func_l = sp.lambdify((x_sym, y_sym) + tuple(theta_syms), Vy_sym, modules='numpy')
    lapV_func_l = sp.lambdify((x_sym, y_sym) + tuple(theta_syms), lapV_sym, modules='numpy')

    dDdx_funcs = [sp.lambdify((x_sym, y_sym), g[0], modules='numpy') for g in gradD_syms]
    dDdy_funcs = [sp.lambdify((x_sym, y_sym), g[1], modules='numpy') for g in gradD_syms]
    lapD_funcs = [sp.lambdify((x_sym, y_sym), L, modules='numpy') for L in lapD_syms]

    def D_numeric(X, Y):
        return np.stack([f(X, Y) for f in D_funcs], axis=-1)

    def V_numeric(X, Y, theta):
        return V_func_l(X, Y, *tuple(theta))

    def gradV_numeric(X, Y, theta):
        return Vx_func_l(X, Y, *tuple(theta)), Vy_func_l(X, Y, *tuple(theta))

    def lapV_numeric(X, Y, theta):
        return lapV_func_l(X, Y, *tuple(theta))

    def gradientD_numeric(X, Y):
        comps = []
        for dx, dy in zip(dDdx_funcs, dDdy_funcs):
            a = np.asarray(dx(X, Y))
            b = np.asarray(dy(X, Y))
            a, b = np.broadcast_arrays(a, b)
            comps.append(np.stack([a, b], axis=-1))
        # stack components into (..., nD, 2)
        return np.stack(comps, axis=-2)

    def laplacianD_numeric(X, Y):
        # ensure each component is broadcast to the grid shape
        grid_shape = np.asarray(X).shape
        vals = [np.broadcast_to(np.asarray(f(X, Y)), grid_shape) for f in lapD_funcs]
        return np.stack(vals, axis=-1)

    # also return symbolic objects for user inspection
    return {
        'D_func': D_numeric,
        'V_func': V_numeric,
        'gradV_func': gradV_numeric,
        'lapV_func': lapV_numeric,
        'gradientD_func': gradientD_numeric,
        'laplacianD_func': laplacianD_numeric,
        'V_sym': V_sym,
        'Vx_sym': Vx_sym,
        'Vy_sym': Vy_sym,
        'lapV_sym': lapV_sym,
        'gradD_syms': gradD_syms,
        'lapD_syms': lapD_syms,
    }
def weighted_diffusion_system(M_field: Array2, mask_dirichlet_inner: Array2, dx: float, dy: float):
    nx, ny = M_field.shape

    def k(i, j):
        return i * ny + j

    A = sp.lil_matrix((nx * ny, nx * ny))
    rhs = np.zeros(nx * ny)

    for i in range(nx):
        for j in range(ny):
            idx = k(i, j)

            if mask_dirichlet_inner[i, j]:
                A[idx, idx] = 1.0
                rhs[idx] = 0.0
                continue

            if i == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i + 1, j)] = -1.0
                continue
            if i == nx - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i - 1, j)] = -1.0
                continue
            if j == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j + 1)] = -1.0
                continue
            if j == ny - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j - 1)] = -1.0
                continue

            mxp = 0.5 * (M_field[i, j] + M_field[i + 1, j])
            mxm = 0.5 * (M_field[i, j] + M_field[i - 1, j])
            myp = 0.5 * (M_field[i, j] + M_field[i, j + 1])
            mym = 0.5 * (M_field[i, j] + M_field[i, j - 1])

            cxp = mxp / dx ** 2
            cxm = mxm / dx ** 2
            cyp = myp / dy ** 2
            cym = mym / dy ** 2
            c = -(cxp + cxm + cyp + cym)

            A[idx, k(i, j)] = c
            A[idx, k(i + 1, j)] = cxp
            A[idx, k(i - 1, j)] = cxm
            A[idx, k(i, j + 1)] = cyp
            A[idx, k(i, j - 1)] = cym

    return A.tocsr(), rhs



def estimate_D_on_boundary(X: Array2, Y: Array2, mask_P: Array2, D_func: Callable[[Array2, Array2], Array2] = default_D):
    D_vals = D_func(X, Y)
    if np.any(mask_P):
        return D_vals[mask_P].mean(axis=0)
    return D_vals.mean(axis=(0, 1))


def solve_committor_upwind(nx, ny, x_range, y_range, theta, gradV_func, beta=1.0, gamma=1.0, m=1.0, radius=0.1):
    X, Y, dx, dy = make_grid(x_range, y_range, nx, ny)
    dist_A = np.sqrt((X + 1) ** 2 + Y ** 2)
    dist_B = np.sqrt((X - 1) ** 2 + Y ** 2)
    mask_A = dist_A <= radius
    mask_B = dist_B <= radius

    Vx, Vy = gradV_func(X, Y, theta)
    inv_beta = 1.0 / beta
    inv_fric = 1.0 / (gamma * m)

    def k(i, j):
        return i * ny + j

    A = sp.lil_matrix((nx * ny, nx * ny))
    rhs = np.zeros(nx * ny)

    for i in range(nx):
        for j in range(ny):
            idx = k(i, j)

            if mask_A[i, j]:
                A[idx, idx] = 1.0
                rhs[idx] = 0.0
                continue
            if mask_B[i, j]:
                A[idx, idx] = 1.0
                rhs[idx] = 1.0
                continue

            if i == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i + 1, j)] = -1.0
                continue
            if i == nx - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i - 1, j)] = -1.0
                continue
            if j == 0:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j + 1)] = -1.0
                continue
            if j == ny - 1:
                A[idx, k(i, j)] = 1.0
                A[idx, k(i, j - 1)] = -1.0
                continue

            c = -2.0 * inv_beta * inv_fric / dx ** 2 - 2.0 * inv_beta * inv_fric / dy ** 2
            cxm = inv_beta * inv_fric / dx ** 2
            cxp = inv_beta * inv_fric / dx ** 2
            cym = inv_beta * inv_fric / dy ** 2
            cyp = inv_beta * inv_fric / dy ** 2

            vx = Vx[i, j] * inv_fric
            vy = Vy[i, j] * inv_fric

            if vx >= 0.0:
                c += -vx / dx
                cxm += vx / dx
            else:
                c += vx / dx
                cxp += -vx / dx

            if vy >= 0.0:
                c += -vy / dy
                cym += vy / dy
            else:
                c += vy / dy
                cyp += -vy / dy

            A[idx, k(i, j)] = c
            A[idx, k(i - 1, j)] = cxm
            A[idx, k(i + 1, j)] = cxp
            A[idx, k(i, j - 1)] = cym
            A[idx, k(i, j + 1)] = cyp

    q_flat = spsolve(A.tocsr(), rhs)
    return q_flat.reshape((nx, ny))


def g(x, laplacianD_func, gradientD_func, gradientV_func, beta, gamma=1.0, m=1.0):
    inv_fric = 1.0 / (gamma * m)
    return inv_fric * (0.5 * laplacianD_func(x[0], x[1]) - 0.5 * beta * (
        gradientD_func(x[0], x[1]) @ gradientV_func(x[0], x[1], np.array([1.0, 1.0]))
    ))

