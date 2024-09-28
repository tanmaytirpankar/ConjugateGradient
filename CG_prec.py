import sys
from fileinput import filename

import numpy as np
import matplotlib.pyplot as plt


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


def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Solves the system A x = b using the Conjugate Gradient method and collects data per iteration.

    Parameters:
    A : ndarray
        Coefficient matrix (must be symmetric positive definite)
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess for the solution
    tol : float, optional
        Tolerance for stopping criterion
    max_iter : int, optional
        Maximum number of iterations

    Returns:
    x : ndarray
        Approximate solution to A x = b
    data : dict
        Dictionary of collected data per iteration (direction vector, angular change, residuals, etc.)
    """

    # Ensure x0 is a NumPy array
    x = np.array(x0)
    r = b - A @ x  # Residual
    p = r.copy()  # Search direction
    residual_norm = np.linalg.norm(r)
    history = {'direction': [], 'angular_change': [], 'residual': [], 'delta_residual': [], 'min_x': [], 'max_x': []}

    r_prev = r.copy()  # Store previous residual for residual change calculation

    # Print the initial matrix and vectors
    print("Initial matrix A:")
    print(A)
    print("Initial vector b:")
    print(b)
    print("Initial guess x0:")
    print(x0)
    print("Condition number of A:", np.linalg.cond(A))

    for k in range(max_iter):
        # Store current values
        history['direction'].append(p.copy())
        history['residual'].append(residual_norm)

        try:
            # Append min and max values of x
            history['min_x'].append(np.min(x))
            history['max_x'].append(np.max(x))
        except ValueError:
            print(f"Error with x at iteration {k}: {x}")
            break  # If there is an issue with the vector x

        # Print matrix and vectors at each iteration
        print(f"\nIteration {k + 1}:")
        print(f"  Solution vector x: {x}")
        print(f"  Residual vector r: {r}")
        print(f"  Residual norm: {residual_norm}")
        print(f"  Search direction p: {p}")

        if residual_norm < tol:
            print(f"Converged at iteration {k}")
            break

        # Conjugate Gradient formula: compute step size alpha
        Ap = A @ p
        alpha = (r.T @ r) / (p.T @ Ap)

        # Update solution
        x = x + alpha * p

        # Compute new residual
        r_new = r - alpha * Ap
        residual_norm_new = np.linalg.norm(r_new)
        delta_residual = residual_norm_new - residual_norm
        history['delta_residual'].append(delta_residual)

        # Compute angular change between current and previous direction vectors
        if k > 0:
            cos_theta = np.dot(p, history['direction'][k - 1]) / (
                        np.linalg.norm(p) * np.linalg.norm(history['direction'][k - 1]))
            angular_change = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # in degrees
            history['angular_change'].append(angular_change)
        else:
            history['angular_change'].append(0)  # No angular change at first iteration

        # Check convergence
        if residual_norm_new < tol:
            print(f"Converged at iteration {k}")
            break

        # Conjugate Gradient component: update search direction using beta
        beta = (r_new.T @ r_new) / (r.T @ r)
        p = r_new + beta * p

        # Update residual for the next iteration
        r = r_new
        residual_norm = residual_norm_new

        print(f"  Delta residual: {history['delta_residual'][-1]}")
        print(f"  Angular change: {history['angular_change'][-1]} degrees")
        print(f"  Min x: {history['min_x'][-1]}")
        print(f"  Max x: {history['max_x'][-1]}")

    print(f"  Delta residual: {history['delta_residual'][-1]}")
    print(f"  Angular change: {history['angular_change'][-1]} degrees")
    print(f"  Min x: {history['min_x'][-1]}")
    print(f"  Max x: {history['max_x'][-1]}")

    # Ensure that all lists in history are the same length before returning
    min_len = min(len(history['residual']), len(history['delta_residual']), len(history['min_x']),
                  len(history['max_x']), len(history['angular_change']))

    # Truncate lists to min length if necessary
    for key in history:
        history[key] = history[key][:min_len]

    return x, history


def plot_conjugate_gradient_data(history):
    """
    Plots the collected data from the Conjugate Gradient method using matplotlib.

    Parameters:
    history : dict
        Dictionary containing the conjugate gradient data per iteration.
    """
    iterations = range(len(history['residual']))  # Ensure all lists are the same length

    # Plot residual and delta_residual
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(iterations, history['residual'], label='Residual')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Residual vs Iterations')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(iterations, history['delta_residual'], label='Change in Residual', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Change in Residual')
    plt.title('Change in Residual vs Iterations')
    plt.grid(True)

    # Plot angular change
    plt.subplot(2, 2, 3)
    plt.plot(iterations, history['angular_change'], label='Angular Change (Degrees)', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Angular Change (Degrees)')
    plt.title('Angular Change vs Iterations')
    plt.grid(True)

    # Plot min and max values of x
    plt.subplot(2, 2, 4)
    plt.plot(iterations, history['min_x'], label='Min x', color='blue')
    plt.plot(iterations, history['max_x'], label='Max x', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Value of x')
    plt.title('Min and Max x Values vs Iterations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Read input filename from first command line argument
    filename = sys.argv[1]

    # Read matrix A, vector b, and initial guess x0 from file
    A, b, x0 = read_matrix_and_vectors_from_file(filename)

    # Solve using Conjugate Gradient method
    solution, history = conjugate_gradient(A, b, x0, tol=1e-6, max_iter=100)

    # Plot the data
    plot_conjugate_gradient_data(history)
