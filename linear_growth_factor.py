import numpy as np
from scipy.integrate import quad

def linear_growth_factor(z, Omega_m, Omega_Lambda, H0=70.0, z_max=1000.0):
    """
    Compute the normalized linear growth factor D(z), normalized such that D(z=0) = 1

    Parameters:
    z : float or array-like
        Target redshift value(s) (single value or array)
    Omega_m : float
        Matter density parameter
    Omega_Lambda : float
        Dark energy density parameter
    H0 : float, optional
        Hubble constant (km/s/Mpc), default is 70.0
    z_max : float, optional
        Upper integration limit (used instead of infinity), default is 1000.0

    Returns:
    float or ndarray
        Normalized linear growth factor D(z)

    Example:
    >>> # Compute growth factor at a single redshift
    >>> d_z1 = linear_growth_factor(1.0, 0.3, 0.7)
    >>> 
    >>> # Compute growth factors at multiple redshifts
    >>> z_arr = [0, 0.5, 1.0, 2.0, 3.0]
    >>> d_arr = linear_growth_factor(z_arr, 0.3, 0.7)
    """
    # Inner function: compute Hubble parameter H(z)
    def H(z_val):
        return H0 * np.sqrt(Omega_m * (1 + z_val)**3 + 
                            Omega_Lambda + 
                            (1 - Omega_m - Omega_Lambda) * (1 + z_val)**2)
    
    # Inner function: integrand (1+z)/H(z)^3
    def integrand(z_prime):
        return (1 + z_prime) / H(z_prime)**3
    
    # Compute unnormalized D(z) at a single redshift
    def compute_D(z_val):
        integral, _ = quad(integrand, z_val, z_max)
        return H(z_val) * integral
    
    # Compute normalization constant D(z=0)
    D0 = compute_D(0.0)
    
    # Handle scalar or array input
    if np.isscalar(z):
        return compute_D(z) / D0
    else:
        # Compute D(z) for each redshift
        return np.array([compute_D(zi) / D0 for zi in z])


# Example usage and plotting function
def plot_growth_factors(models, z_range=(0, 10), num_points=100, log_scale=False):
    """
    Plot the evolution of the linear growth factor for different cosmological models

    Parameters:
    models : list of dict
        List of cosmological models, each dictionary should contain:
        {'Omega_m': float, 'Omega_Lambda': float, 'label': str, 'color': str}
    z_range : tuple, optional
        Redshift range (start, end), default is (0, 10)
    num_points : int, optional
        Number of redshift points, default is 100
    log_scale : bool, optional
        Whether to use logarithmic axes, default is False
    """
    import matplotlib.pyplot as plt
    
    # Create redshift array
    z_values = np.linspace(z_range[0], z_range[1], num_points)
    
    plt.figure(figsize=(10, 7))
    
    # Compute and plot D(z) for each model
    for model in models:
        Omega_m = model["Omega_m"]
        Omega_Lambda = model["Omega_Lambda"]
        label = model.get("label", f"Ωm={Omega_m}, ΩΛ={Omega_Lambda}")
        color = model.get("color", None)
        
        # Compute growth factor
        D_values = linear_growth_factor(z_values, Omega_m, Omega_Lambda)
        
        # Plot result
        if log_scale:
            plt.plot(z_values + 1, D_values, label=label, color=color, linewidth=2.5)
        else:
            plt.plot(z_values, D_values, label=label, color=color, linewidth=2.5)
    
    # Set plot labels and title
    plt.xlabel('1 + z' if log_scale else 'Redshift (z)', fontsize=14)
    plt.ylabel('D(z)', fontsize=14)
    
    title = 'Linear Growth Factor'
    title += ' (Log Scale)' if log_scale else '(Linear Scale)'
    plt.title(title, fontsize=16)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc='lower left' if log_scale else 'upper right')
    plt.tight_layout()
    
    # Save figures
    suffix = "_log" if log_scale else "_linear"
    plt.savefig(f"growth_factor{suffix}.pdf", format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example cosmological models
    models = [
        {
            "Omega_m": 1.0, 
            "Omega_Lambda": 0.0, 
            "label": r"$\Omega_m=1, \Omega_\Lambda=0$ (Einstein-de Sitter)", 
            "color": "blue"
        },
        {
            "Omega_m": 0.3, 
            "Omega_Lambda": 0.7, 
            "label": r"$\Omega_m=0.3, \Omega_\Lambda=0.7$ (ΛCDM)", 
            "color": "red"
        },
        {
            "Omega_m": 0.3, 
            "Omega_Lambda": 0.0, 
            "label": r"$\Omega_m=0.3, \Omega_\Lambda=0$ (Matter-dominated)", 
            "color": "green"
        },
        {
            "Omega_m": 0.25, 
            "Omega_Lambda": 0.75, 
            "label": r"$\Omega_m=0.25, \Omega_\Lambda=0.75$", 
            "color": "purple"
        }
    ]
    
    # Compute sample values
    print("Example calculations:")
    print(f"ΛCDM model at z=0: D(z) = {linear_growth_factor(0.0, 0.3, 0.7):.4f}")
    print(f"ΛCDM model at z=1: D(z) = {linear_growth_factor(1.0, 0.3, 0.7):.4f}")
    print(f"ΛCDM model at z=2: D(z) = {linear_growth_factor(2.0, 0.3, 0.7):.4f}")
    
    # Plot linear scale
    plot_growth_factors(models, log_scale=False)
    
    # Plot logarithmic scale
    plot_growth_factors(models, log_scale=True)
