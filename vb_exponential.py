import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm

class VariationalInference2DGaussian:
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.q_z1 = {'mean': -2, 'var': 1}
        self.q_z2 = {'mean': -1.5, 'var': 3}
        
    def update_q_z1(self):

        # Update q(z1) using the formula: μ1 + (σ12/σ22) * (E[z2] - μ2)
        sigma_12 = self.Sigma[0, 1]
        sigma_22 = self.Sigma[1, 1]
        
        # E[z2] is the mean of q(z2)
        E_z2 = self.q_z2['mean']
        
        # Calculate the updated mean for q(z1)
        updated_mean = self.mu[0] + (sigma_12 / sigma_22) * (E_z2 - self.mu[1])
        
        # Update the variance (this remains the same as before)
        updated_var = self.Sigma[0, 0] - (sigma_12**2 / sigma_22)
        
        # Set the new mean and variance for q(z1)
        self.q_z1['mean'] = updated_mean
        self.q_z1['var'] = updated_var


        
    def update_q_z2(self):
        # Update q(z2) using the formula: μ2 + (σ21/σ11) * (E[z1] - μ1)
        sigma_21 = self.Sigma[1, 0]
        sigma_11 = self.Sigma[0, 0]
        
        # E[z1] is the mean of q(z1)
        E_z1 = self.q_z1['mean']
        
        # Calculate the updated mean for q(z2)
        updated_mean = self.mu[1] + (sigma_21 / sigma_11) * (E_z1 - self.mu[0])
        
        # Update the variance
        updated_var = self.Sigma[1, 1] - (sigma_21**2 / sigma_11)
        
        # Set the new mean and variance for q(z2)
        self.q_z2['mean'] = updated_mean
        self.q_z2['var'] = updated_var
        
    def iterate(self):
        self.update_q_z1()
        self.update_q_z2()

# Example usage
mu = np.array([1, 2])
Sigma = np.array([[2, 0.5], [0.5, 1]])

vi = VariationalInference2DGaussian(mu, Sigma)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

# True distribution
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z_true = multivariate_gaussian(pos, mu, Sigma)

# Initial variational distribution
Z_q = np.zeros_like(Z_true)

im = ax.imshow(Z_true, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis', alpha=0.5)
contour_true = ax.contour(X, Y, Z_true, colors='k', alpha=0.5)
contour_q = ax.contour(X, Y, Z_q, colors='r', alpha=0.5)

ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_title('True Distribution (black) vs Variational Distribution (red)')

# Add marginal distributions
ax_marg_x = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
ax_marg_y = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)

line_q1, = ax_marg_x.plot([], [], 'r-', lw=2, label='q(z1)')
line_q2, = ax_marg_y.plot([], [], 'r-', lw=2, label='q(z2)')

ax_marg_x.set_ylim(0, 0.5)
ax_marg_y.set_xlim(0, 0.5)

ax_marg_x.axis('off')
ax_marg_y.axis('off')

def update_plot(event):
    vi.iterate()
    global contour_q, line_q1, line_q2
    
    # Update variational distribution
    Z_q = multivariate_gaussian(pos, np.array([vi.q_z1['mean'], vi.q_z2['mean']]), 
                                np.diag([vi.q_z1['var'], vi.q_z2['var']]))
    
    # Clear previous contours and plot new ones
    for coll in contour_q.collections:
        coll.remove()
    
    contour_q = ax.contour(X, Y, Z_q, colors='r', alpha=0.5)
    
    # Update marginal distributions
    z1_range = np.linspace(-5, 5, 100)
    z2_range = np.linspace(-5, 5, 100)
    
    q1_pdf = norm.pdf(z1_range, vi.q_z1['mean'], np.sqrt(vi.q_z1['var']))
    q2_pdf = norm.pdf(z2_range, vi.q_z2['mean'], np.sqrt(vi.q_z2['var']))
    
    line_q1.set_data(z1_range, q1_pdf)
    line_q2.set_data(q2_pdf, z2_range)
    
    ax_marg_x.clear()
    ax_marg_y.clear()
    ax_marg_x.plot(z1_range, q1_pdf, 'r-', lw=2)
    ax_marg_y.plot(q2_pdf, z2_range, 'r-', lw=2)
    ax_marg_x.set_ylim(0, max(q1_pdf) * 1.1)
    ax_marg_y.set_xlim(0, max(q2_pdf) * 1.1)
    ax_marg_x.axis('off')
    ax_marg_y.axis('off')
    
    fig.canvas.draw_idle()

# Add a button for iteration
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Iterate')
button.on_clicked(update_plot)

plt.show()