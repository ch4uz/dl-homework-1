import numpy as np
import matplotlib.pyplot as plt

# Define t range
t = np.linspace(-4, 4, 1000)

# Parameter for relumax
b = 1.0

# =============================================================================
# Function Definitions
# =============================================================================

def softmax_2(t):
    """Softmax second component: sigmoid function"""
    return 1 / (1 + np.exp(-t))

def softmax_2_derivative(t):
    """Derivative of softmax second component"""
    s = softmax_2(t)
    return s * (1 - s)

def sparsemax_2(t):
    """Sparsemax second component"""
    return np.clip((1 + t) / 2, 0, 1)

def sparsemax_2_derivative(t):
    """Derivative of sparsemax second component"""
    return np.where((t > -1) & (t < 1), 0.5, 0.0)

def relumax_2(t, b):
    """Relumax second component"""
    result = np.zeros_like(t)

    # Region: t <= -b
    mask1 = t <= -b
    result[mask1] = 0

    # Region: -b < t <= 0
    mask2 = (t > -b) & (t <= 0)
    result[mask2] = (t[mask2] + b) / (2*b + t[mask2])

    # Region: 0 < t < b
    mask3 = (t > 0) & (t < b)
    result[mask3] = b / (2*b - t[mask3])

    # Region: t >= b
    mask4 = t >= b
    result[mask4] = 1

    return result

def relumax_2_derivative(t, b):
    """Derivative of relumax second component"""
    result = np.zeros_like(t)

    # Region: -b < t < 0
    mask2 = (t > -b) & (t < 0)
    result[mask2] = b / (2*b + t[mask2])**2

    # Region: 0 < t < b
    mask3 = (t > 0) & (t < b)
    result[mask3] = b / (2*b - t[mask3])**2

    # At t = 0, derivative is 1/(4b)
    mask0 = np.abs(t) < 1e-10
    result[mask0] = 1 / (4*b)

    return result

# =============================================================================
# Plotting
# =============================================================================

# Plot 1: Activation Functions
fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111)
ax1.plot(t, softmax_2(t), 'b-', linewidth=2, label='softmax')
ax1.plot(t, sparsemax_2(t), 'r-', linewidth=2, label='sparsemax')
ax1.plot(t, relumax_2(t, b), 'g-', linewidth=2, label=f'relumax (b={b})')

ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axhline(y=1, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.axvline(x=-1, color='r', linewidth=0.5, linestyle=':', alpha=0.5)
ax1.axvline(x=1, color='r', linewidth=0.5, linestyle=':', alpha=0.5)
ax1.axvline(x=-b, color='g', linewidth=0.5, linestyle=':', alpha=0.5)
ax1.axvline(x=b, color='g', linewidth=0.5, linestyle=':', alpha=0.5)

ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel('Activation value', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('q3/plots/activation_functions.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# Plot 2: Derivatives
fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(t, softmax_2_derivative(t), 'b-', linewidth=2, label='softmax derivative')
ax2.plot(t, sparsemax_2_derivative(t), 'r-', linewidth=2, label='sparsemax derivative')
ax2.plot(t, relumax_2_derivative(t, b), 'g-', linewidth=2, label=f'relumax derivative (b={b})')

ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.axvline(x=-1, color='r', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.axvline(x=1, color='r', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.axvline(x=-b, color='g', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.axvline(x=b, color='g', linewidth=0.5, linestyle=':', alpha=0.5)

ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('Derivative value', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('q3/plots/activation_derivatives.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

print("Plots saved as 'activation_functions.png' and 'activation_derivatives.png'")
