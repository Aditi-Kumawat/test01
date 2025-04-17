import numpy as np

def get_w0_t_vect_Newton_I_TPV_zt(w1_t_vect, t_vect, t_itr, I_TP_V, C_H, z_t_vect):
    """
    Newton-Raphson to find roots of non-linear algebraic equations (for nonlinear Hertzian contact
    including the effect of track irregularity).

    Parameters:
        w1_t_vect (array-like): Input vector corresponding to w1.
        t_vect (array-like): Time or spatial vector.
        t_itr (int): Maximum number of iterations.
        I_TP_V (float): Constant used in the formulation.
        C_H (float): Constant used in the formulation.
        z_t_vect (array-like): Input vector corresponding to track irregularity z_t.

    Returns:
        w0_t_vect (np.ndarray): Computed solution vector for w0.
        t_itr_I_end (int): The number of iterations performed before convergence.
    """
    # Convert inputs to NumPy arrays
    w1 = np.array(w1_t_vect)
    z = np.array(z_t_vect)
    t = np.array(t_vect)
    n = len(t)
    
    # Initialize storage for iterative solutions
    w_0_t_itr_mat = np.zeros((n, t_itr + 1))
    
    # Create a list of indices for the unconverged entries
    idx_vect = list(range(n))
    
    # Initial guess (column vector in MATLAB becomes a 1D array in Python)
    w0_t_vect_T = np.ones(n) * 1e-5
    
    # Pre-calculate constant factor
    pi_factor = (np.pi / I_TP_V / C_H) ** (2/3)
    
    t_itr_I_end = t_itr  # Default iteration count if maximum iterations are reached
    
    # Newton-Raphson iterative process
    for t_itr_I in range(t_itr):
        # Compute the Newton update for the indices in idx_vect
        # f(w0) = w0 + pi_factor*w0^(2/3) - w1 + z, and
        # f'(w0) = 1 + (pi_factor)*(2/3)*(1/w0^(1/3))
        num = w0_t_vect_T[idx_vect] + pi_factor * (w0_t_vect_T[idx_vect] ** (2/3)) - w1[idx_vect] + z[idx_vect]
        den = 1 + pi_factor * (2/3) * (1 / (w0_t_vect_T[idx_vect] ** (1/3)))
        new_values = w0_t_vect_T[idx_vect] - num / den
        
        # Store the new values in the iteration matrix (as a column)
        w_0_t_itr_mat[np.ix_(idx_vect, [t_itr_I + 1])] = new_values.reshape(-1, 1)
        
        # Update the current guess (ensure it is real)
        w0_t_vect_T[idx_vect] = np.real(new_values)
        
        # Check for convergence: if the change is less than 1e-20 for each index, mark it as converged.
        diff = np.abs(np.real(w_0_t_itr_mat[idx_vect, t_itr_I + 1]) - 
                      np.real(w_0_t_itr_mat[idx_vect, t_itr_I]))
        # Keep only indices that have not yet converged
        new_idx_vect = [idx for j, idx in enumerate(idx_vect) if diff[j] >= 1e-20]
        
        # Update the iteration count and indices for unconverged entries
        t_itr_I_end = t_itr_I + 1
        if not new_idx_vect:
            # All indices have converged
            break
        else:
            idx_vect = new_idx_vect
    
    # Final solution: return the real part of the solution vector
    w0_t_vect = np.real(w0_t_vect_T)
    return w0_t_vect, t_itr_I_end

# Example usage:
if __name__ == "__main__":
    # Sample inputs for testing the function:
    w1 = np.linspace(1, 2, 10)
    t = np.linspace(0, 1, 10)
    z = np.linspace(0.1, 0.2, 10)
    max_iter = 50
    I_TP_V = 1.0
    C_H = 1.0
    
    w0, iterations = get_w0_t_vect_Newton_I_TPV_zt(w1, t, max_iter, I_TP_V, C_H, z)
    print("w0_t_vect:", w0)
    print("Iterations performed:", iterations)
