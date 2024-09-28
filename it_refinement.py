import sys

import numpy as np
from numpy.linalg import solve, norm
from scipy.linalg import solve as precise_solve

def read_matrix_and_vectors_from_file(filename):
    """
    Reads matrix A, vector b, and initial guess x0 from a file.

    Parameters:
    filename : str
        The path to the file containing the matrix and vectors.

    Returns:
    A : ndarray
        Coefficient matrix A.
    b : ndarray
        Right-hand side vector b.
    x0 : ndarray
        Initial guess vector x0.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    matrix_section = []
    vector_b_section = []
    vector_x0_section = []

    current_section = None

    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            continue  # Ignore comment and empty lines

        # Determine which section we are in
        if "Matrix A" in line:
            current_section = 'matrix'
        elif "Vector b" in line:
            current_section = 'vector_b'
        elif "Initial guess x0" in line:
            current_section = 'vector_x0'
        else:
            # Add the current line to the appropriate section
            if current_section == 'matrix':
                matrix_section.append([float(x) for x in stripped_line.split(",")])
            elif current_section == 'vector_b':
                vector_b_section.append(float(stripped_line))
            elif current_section == 'vector_x0':
                vector_x0_section.append(float(stripped_line))

    A = np.array(matrix_section)
    b = np.array(vector_b_section)
    x0 = np.array(vector_x0_section)

    return A, b, x0


def iterative_refinement(A, b, x0, tol=1e-10, max_iterations=1000, u1=1e-8, u2=1e-10, u3=1e-12):
    """
    Solves the system Ax = b using an iterative refinement technique with variable precision.

    Parameters:
    A : numpy.ndarray
        Coefficient matrix
    b : numpy.ndarray
        Right-hand side vector
    x0 : numpy.ndarray
        Initial guess for the solution
    tol : float
        Tolerance for convergence
    max_iterations : int
        Maximum number of iterations
    u1 : float
        Precision for the first solve
    u2 : float
        Precision for computing residuals and updating x
    u3 : float
        Precision for solving correction equation

    Returns:
    x : numpy.ndarray
        Approximate solution vector
    k : int
        Number of iterations
    """

    def solve_with_precision(A, b, precision):
        """ Helper function to solve the system A * x = b with a specified precision. """
        return precise_solve(A, b, assume_a='pos', check_finite=False).astype(
            np.float64 if precision < 1e-9 else np.float32)

    def compute_residual(A, x, b, precision):
        """ Compute the residual r = b − Ax with a specified precision. """
        return (b - np.dot(A, x)).astype(np.float64 if precision < 1e-9 else np.float32)

    # Set the initial guess for x
    x = x0.copy().astype(np.float64 if u1 < 1e-9 else np.float32)

    print("Condition number of A:", np.linalg.cond(A))

    # To collect residuals
    residuals = []

    # Compute residual: r = b − Ax @ precision u2
    r = compute_residual(A, x, b, u2)
    # Store the residual norm
    residual_norm = np.linalg.norm(r, ord=np.inf)
    residuals.append(residual_norm)
    print(f'Residual: {residual_norm}')
    print(f"  Residual vector r: {r}")

    for k in range(max_iterations):
        # Solve: A * x = b @ precision u1 (initial solve)
        x_new = solve_with_precision(A, b, u1)

        # Compute residual: r = b − Ax @ precision u2
        r = compute_residual(A, x_new, b, u2)

        # Store the residual norm
        residual_norm = np.linalg.norm(r, ord=np.inf)
        residuals.append(residual_norm)
        print(f'Iteration {k + 1}, x: {x_new}')
        print(f'Residual: {residual_norm}')
        print(f"  Residual vector r: {r}")

        # Check if the residual is small enough (convergence condition)
        if residual_norm < tol:
            print(f'Converged in {k + 1} iterations.')
            return x_new, k + 1, residuals

        # Solve: Ae = r @ precision u3 (solve correction equation)
        e = solve_with_precision(A, r, u3)

        # Update solution: x = x + e @ precision u2
        x = (x_new + e).astype(np.float64 if u2 < 1e-9 else np.float32)

        # Compute new residual to check if we've converged after the update
        r_new = compute_residual(A, x, b, u2)
        residual_norm_new = np.linalg.norm(r_new, ord=np.inf)
        residuals.append(residual_norm_new)
        print(f'Update after correction, Residual: {residual_norm_new}')

        if residual_norm_new < tol:
            print(f'Converged in {k + 1} iterations.')
            return x, k + 1, residuals

    print(f'Max iterations reached ({max_iterations})')
    return x, max_iterations, residuals


# Example usage:
if __name__ == "__main__":
    # Read input filename from first command line argument
    filename = sys.argv[1]

    # Read matrix A, vector b, and initial guess x0 from file
    A, b, x0 = read_matrix_and_vectors_from_file(filename)

    # Call the function with desired precisions
    x_approx, iterations, residuals = iterative_refinement(A, b, x0, u1=1e-8, u2=1e-10, u3=1e-8)

    print(f'Approximate solution: {x_approx}')
    print(f'Number of iterations: {iterations}')
    print(f'Residuals at each iteration: {residuals}')
