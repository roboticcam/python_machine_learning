import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm, gamma

def true_posterior(mu, tau, data, mu0, lambda0, a0, b0):
    N = len(data)
    x_bar = np.mean(data)
    
    lambda_n = lambda0 + N
    mu_n = (lambda0 * mu0 + N * x_bar) / lambda_n
    a_n = a0 + N / 2
    b_n = b0 + 0.5 * (np.sum((data - x_bar)**2) + 
                      N * lambda0 / lambda_n * (x_bar - mu0)**2)
    return norm.pdf(mu, mu_n, 1/np.sqrt(lambda_n*tau)) * gamma.pdf(tau, a_n, scale=1/b_n)

class VariationalInferenceNormalGamma:
    def __init__(self, data, mu0, lambda0, a0, b0):
        self.data = data
        self.N = len(data)
        self.x_bar = np.mean(data)
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.a0 = a0
        self.b0 = b0
        
        # Initialize variational parameters - far from expected valueså
        self.lambda_n = 0.1  # Much smaller than lambda0 + N
        self.mu_n = 10.0  # Far from the expected mean
        self.a_n = 0.5  # Much smaller than a0 + N/2
        self.b_n = 10.0  # Much larger than expected
   
    
    def update(self):
        self.lambda_n = self.lambda0 + self.N
        self.mu_n = (self.lambda0 * self.mu0 + self.N * self.x_bar) / self.lambda_n
        self.a_n = self.a0 + self.N / 2
        E_tau = self.a_n / self.b_n
        E_mu2 = 1 / (E_tau * self.lambda_n) + self.mu_n**2
        self.b_n = self.b0 + 0.5 * (np.sum(self.data**2) + 
                                    self.lambda0 * self.mu0**2 + 
                                    (self.lambda_n * E_mu2) - 
                                    2 * (self.N * self.x_bar + self.lambda0 * self.mu0) * self.mu_n)

    def plot(self, ax_joint, ax_marg_x, ax_marg_y):
        # Plot joint distribution
        mu_range = np.linspace(self.mu_n - 3/np.sqrt(self.lambda_n), self.mu_n + 3/np.sqrt(self.lambda_n), 100)
        tau_range = np.linspace(0, self.a_n/self.b_n*3, 100)
        Mu, Tau = np.meshgrid(mu_range, tau_range)
        
        # Variational distribution
        Z_var = np.exp(norm.logpdf(Mu, self.mu_n, 1/np.sqrt(self.lambda_n*Tau)) + 
                       gamma.logpdf(Tau, self.a_n, scale=1/self.b_n))
        
        # True posterior
        Z_true = true_posterior(Mu, Tau, self.data, self.mu0, self.lambda0, self.a0, self.b0)
        
        # Plot both distributions
        ax_joint.contourf(Mu, Tau, Z_var, levels=20, alpha=0.5, cmap='Blues')
        ax_joint.contour(Mu, Tau, Z_true, levels=20, colors='r', alpha=0.5)
        ax_joint.set_xlabel('μ')
        ax_joint.set_ylabel('τ')
        ax_joint.set_title('q(μ,τ) (blue) vs True Posterior (red)')

        # Plot q(μ)
        ax_marg_x.plot(mu_range, norm.pdf(mu_range, self.mu_n, 1/np.sqrt(self.lambda_n * self.a_n/self.b_n)), 'b-', label='q(μ)')
        ax_marg_x.plot(mu_range, norm.pdf(mu_range, self.mu_n, 1/np.sqrt(self.lambda_n * (self.a_n-1)/self.b_n)), 'r--', label='True p(μ)')
        ax_marg_x.set_xlabel('μ')
        ax_marg_x.set_ylabel('Density')
        ax_marg_x.set_title('q(μ) vs True p(μ)')
        ax_marg_x.legend()

        # Plot q(τ)
        ax_marg_y.plot(gamma.pdf(tau_range, self.a_n, scale=1/self.b_n), tau_range, 'b-', label='q(τ)')
        ax_marg_y.plot(gamma.pdf(tau_range, self.a_n-1, scale=1/self.b_n), tau_range, 'r--', label='True p(τ)')
        ax_marg_y.set_xlabel('Density')
        ax_marg_y.set_ylabel('τ')
        ax_marg_y.set_title('q(τ) vs True p(τ)')
        ax_marg_y.legend()

# Generate some example data
np.random.seed(0)
true_mu, true_tau = 1, 2
data = np.random.normal(true_mu, 1/np.sqrt(true_tau), 5)

# Initialize the variational inference object
vi = VariationalInferenceNormalGamma(data, mu0=0, lambda0=1, a0=1, b0=1)

# Set up the plot
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax_joint = fig.add_subplot(gs[1, 0])
ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

# Create a button for updating
ax_button = plt.axes([0.4, 0.02, 0.2, 0.075])
button = Button(ax_button, 'Update')

# Define the update function
def update(event):
    vi.update()
    ax_joint.clear()
    ax_marg_x.clear()
    ax_marg_y.clear()
    vi.plot(ax_joint, ax_marg_x, ax_marg_y)
    fig.canvas.draw()

button.on_clicked(update)

# Initial plot
vi.plot(ax_joint, ax_marg_x, ax_marg_y)

plt.show()