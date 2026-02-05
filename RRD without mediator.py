# =============================================================================
#   CYCLIC VOLTAMMETRY SIMULATION: DIRECT GSH OXIDATION (UNMEDIATED)
# =============================================================================
#
#   Description:
#       Simulation of the direct electrochemical oxidation of Glutathione (GSH)
#       at a macroelectrode without a redox mediator.
#       The model assumes Butler-Volmer kinetics and implicit finite difference.
#
#   Mechanism Modeled:
#       Proton Transfer followed by Electron Transfer followed by 
#       Radical-Radical Dimerization (PT-ET-RRD).
#
#   Abbreviations:
#       RRD  : Radical-Radical Dimerization
#       PT   : Proton Transfer
#       ET   : Electron Transfer
#
# =============================================================================
#   REACTION SCHEME & KINETIC MODEL
# =============================================================================
#
#   1. Chemical Steps (Pre-equilibrium):
#      (PT)   GSH             <-->  GS- + H+
#
#   2. Electrochemical Steps:
#      (ET)   GS- - e-        <-->  GS*
#
#   3. Chemical Steps (Post-ET):
#      (RRD)  2GS* <-->  GSSG
#
# =============================================================================
#   VARIABLE MAPPING & NUMERICAL IMPLEMENTATION
# =============================================================================
#
#   Species Legend:
#   ---------------
#   A : GSH         B : GS-
#   C : GS* D : GSSG
#   H : H+
#
#   Reaction Index & Equations:
#   ---------------------------
#   [0]  A      <-->  B + H        (GSH deprotonation)
#   [1]  B - e- <-->  C            (Oxidation of Thiolate to Radical)
#   [2]  2C     <-->  D            (Radical-Radical Dimerization)
#
# =============================================================================

import numpy as np
import math
import scipy.linalg as la
import time
from numba import njit

start_time = time.time()

# =============================================================================
# 1. PHYSICAL CONSTANTS AND SIMULATION PARAMETERS
# =============================================================================

R = 8.314                           # Gas constant (J mol^-1 K^-1)
T = 298                             # Temperature (K)
F = 96485                           # Faraday constant (C mol^-1)
n_e = 1                             # Number of electrons transferred
SS = 1000                           # Standard state correction factor (1 mol m^-3 = 1000 M)

# Potential Parameters
Ei = +0.03740                       # Initial potential (V)
Ev = +1.23613                       # Vertex potential (V)
Ef = +0.653                         # Formal potential (V)

# Electrode and Scan Settings
scan_rate = 0.1                     # Scan rate (V s^-1)
alpha = 0.75                        # Charge transfer coefficient
epsilon = 1.5E-3                    # Electrode radius (m)
A = (math.pi)*(epsilon**2)          # Electrode area (m^2)

# Solution Conditions
pH = 7                              # Solution pH
pKa = 9                             # pKa of the thiol group
cA_i = 1                            # Initial total concentration of GSH (mol m^-3)
HH_constant = 10**(pH-pKa)          # Henderson-Hasselbalch constant

# =============================================================================
# 2. BULK CONCENTRATIONS (mol m^-3)
# =============================================================================

# Calculate equilibrated species based on pH (Deprotonation Equilibrium)
cB_bulk = cA_i*HH_constant/(HH_constant+1)  # GS-
cA_bulk = cA_i-cB_bulk                      # GSH
cC_bulk = 0                                 # GS*
cD_bulk = 0                                 # GSSG
cH_bulk = (10**(-pH))*1000                  # H+

# =============================================================================
# 3. KINETIC RATE CONSTANTS
# =============================================================================

# Reaction 0: Acid/Base Equilibrium (GSH <-> GS-)
km0 = 1E7                           # Reverse rate constant (m^3 mol^-1 s^-1)
Keq0 = (10**(-pKa))                 # Equilibrium constant
K0_SS = Keq0*SS                     # Standard state corrected Keq
k0 = K0_SS*km0                      # Forward rate constant (s^-1)

# Reaction 1: Electrochemical oxidation (GS- -> GS*)
k1 = 1E-5                           # Standard heterogeneous rate constant (m s^-1)

# Reaction 2: Radical Dimerization (2GS* -> GSSG)
k2 = 1.5E6                          # Forward rate constant (m^3 mol^-1 s^-1)
Keq2 = 9.5E27                       # Equilibrium constant
K2_SS = Keq2/SS                     # Standard state corrected Keq
km2 = k2/K2_SS                      # Reverse rate constant (s^-1)

# =============================================================================
# 4. DIFFUSION COEFFICIENTS (m^2 s^-1)
# =============================================================================

DA = 4.7E-10                        # GSH
DB = 4.7E-10                        # GS-
DC = 4.7E-10                        # GS*
DD = 4.5E-10                        # GSSG

# =============================================================================
# 5. DIMENSIONLESS PARAMETERS (Normalization)
# =============================================================================

# Potential Normalization
theta_i = (F/(R*T))*(Ei-Ef)         # Dimensionless initial potential
theta_v = (F/(R*T))*(Ev-Ef)         # Dimensionless vertex potential
delta_V = 0.1E-2                    # Potential step size (V)
delta_theta = (F/(R*T))*delta_V     # Dimensionless potential step
sigma = ((epsilon**2)*F* # Dimensionless scan rate
         scan_rate/(DA*R*T))

# Dimensionless Kinetic Parameters
K0  = k0 *(epsilon**2)        /DA
Km0 = km0*(epsilon**2)*cA_bulk/DA

K1 = k1*epsilon/DB

K2      = k2*(epsilon**2)*cA_bulk/DA
Km2     = km2*(epsilon**2)      /DA

# Dimensionless Diffusion Coefficients (D_species / D_A)
dA = DA/DA
dB = DB/DA
dC = DC/DA
dD = DD/DA
DC_div_DB = DC/DB                   # Ratio required for flux balance

# Dimensionless Bulk Concentrations (C_species / C_ref)
CA_bulk = cA_bulk/cA_bulk
CB_bulk = cB_bulk/cA_bulk
CC_bulk = cC_bulk/cA_bulk
CD_bulk = cD_bulk/cA_bulk
CH_bulk = cH_bulk/cA_bulk

# =============================================================================
# 6. SPATIAL AND TEMPORAL GRID GENERATION
# =============================================================================

# Temporal Grid
delta_T = delta_theta/sigma         # Dimensionless time step
maxT = 2*abs(theta_v-theta_i)/sigma # Total simulation time (dimensionless)

# Spatial Grid Boundaries
d_max = max(dA, dB, dC, dD)         # Max diffusion coefficient
maxX = 6*math.sqrt(d_max*maxT)      # Outer simulation boundary (6 * diffusion length)

# Exponential Grid Setup
omega = 1.015                       # Grid expansion factor
h = 1e-9                            # Initial spatial step size
X = [0.0, h, 2*h]                   # Initialize Grid Points
x = 2*h
c = 1

# Generate remaining grid points
while x < maxX:
    x += h*(omega**c)
    X.append(x)
    c += 1

# =============================================================================
# 7. PRE-CALCULATION OF KINETIC-TEMPORAL TERMS
# =============================================================================

K0_star  = K0*delta_T
Km0_star = Km0*delta_T

K2_star = K2*delta_T
Km2_star= Km2*delta_T

n = len(X)                          # Number of spatial grid points
m = round(maxT/delta_T)             # Total number of time steps

# Initialize the sparse matrix 'ab' for the band linear solver
ab = np.zeros((9,4*n))
Fx = np.zeros(4*n) 

# =============================================================================
# 8. SOLVER FUNCTIONS (JACOBIAN & RESIDUAL)
# =============================================================================

@njit
def compute_jacobian(n, X, x0, ab, delta_T, dB, dC, dD, K0_star, Km0_star, CH_bulk, K2_star, Km2_star, alpha, Theta, K1, h, DC_div_DB):    
    """
    Constructs the Jacobian matrix for the implicit finite difference solver.
    Populates 'ab' with partial derivatives of the discretized PDEs.
    """
    i = 1  # Index counter for the flattened state vector

    # Loop over internal spatial nodes
    for a in range(1, n-1):
        # Grid steps (Exponential spacing)
        delX_p = X[a+1] - X[a]
        delX_m = X[a] - X[a-1]

        # Pre-calculate diffusion factors (Inverse h terms)
        denom_common_m = delX_m**2 + delX_m*delX_p
        denom_common_p = delX_p**2 + delX_m*delX_p

        inv_hm = 2*delta_T/denom_common_m       # alpha equivalent
        inv_hp = 2*delta_T/denom_common_p       # gamma equivalent

        # Scale geometric factors by diffusion coefficients
        inv_hm_dB = dB*inv_hm; inv_hm_dC = dC*inv_hm; inv_hm_dD = dD*inv_hm
        inv_hp_dB = dB*inv_hp; inv_hp_dC = dC*inv_hp; inv_hp_dD = dD*inv_hp

        # Index Mapping
        i0 = i
        # Unroll indices for 4 species (A, B, C, D) at current node
        i1, i2, i3, i4 = i0+1, i0+2, i0+3, i0+4
        i5, i6, i7, i8 = i0+5, i0+6, i0+7, i0+8
        i9, i10 = i0+9, i0+10
        
        # Current concentration of GS* (Species C, Index 5 in local block)
        x5 = x0[i5]

        # --- Jacobian Elements (Species A: GSH) ---
        ab[8, i0-1] = -inv_hm                                       # Prev Node
        ab[0, i7]   = -inv_hp                                       # Next Node
        ab[4, i3]   = (inv_hm+inv_hp+1) + K0_star                   # Diagonal + Kinetics
        ab[3, i4]   = -Km0_star*CH_bulk                             # Coupling to B

        # --- Jacobian Elements (Species B: GS-) ---
        ab[8, i0]   = -inv_hm_dB
        ab[0, i8]   = -inv_hp_dB
        ab[4, i4]   = (inv_hm_dB+inv_hp_dB+1) + Km0_star*CH_bulk
        ab[5, i3]   = -K0_star                                      # Coupling to A

        # --- Jacobian Elements (Species C: GS*) ---
        ab[8, i1]   = -inv_hm_dC
        ab[0, i9]   = -inv_hp_dC
        ab[4, i5]   = (inv_hm_dC+inv_hp_dC+1) + 2*K2_star*x5        # Diagonal + Dimerization deriv
        ab[3, i6]   = -Km2_star                                     # Coupling to D

        # --- Jacobian Elements (Species D: GSSG) ---
        ab[8, i2]   = -inv_hm_dD
        ab[0, i10]  = -inv_hp_dD
        ab[4, i6]   = (inv_hm_dD+inv_hp_dD+1) + Km2_star/2
        ab[5, i5]   = -K2_star*x5                                   # Coupling to C

        i += 4 # Advance to next spatial node

    # --- Boundary Conditions (Surface x=0) ---
    # Butler-Volmer parameters
    f_BV = math.exp(-alpha*Theta)
    Kred = f_BV*K1 
    Kox = f_BV*K1*math.exp(Theta)

    # Species A (GSH): Zero Flux
    ab[4, 0] = -1
    ab[0, 4] = 1

    # Species B (GS-) and C (GS*): Electroactive Coupling
    # Flux balance coupled with Butler-Volmer rates
    ab[4, 1] = 1+h*Kox
    ab[3, 2] = -h*Kred
    ab[0, 5] = -1

    ab[5, 1] = -(1/DC_div_DB)*h*Kox
    ab[4, 2] = 1+h*Kred*(1/DC_div_DB)
    ab[0, 6] = -1

    # Species D (GSSG): Zero Flux
    ab[4, 3] = -1
    ab[0, 7] = 1

    # --- Boundary Conditions (Bulk x=Inf) ---
    # Dirichlet: C = C_bulk (Identity matrix elements)
    ab[4, 4*n-1] = 1
    ab[4, 4*n-2] = 1
    ab[4, 4*n-3] = 1
    ab[4, 4*n-4] = 1

    return ab

@njit
def compute_Fx(n, x0, ab, Fx, b, K0_star, Km0_star, CH_bulk, K2_star, Km2_star):   
    """
    Computes the Residual Vector -F(x) = J*dx.
    """
    i = 1

    # Loop over internal spatial nodes
    for a in range(1, n-1):
        # Load local concentrations
        x3 = x0[i+3] # A: GSH
        x4 = x0[i+4] # B: GS-
        x5 = x0[i+5] # C: GS*
        x6 = x0[i+6] # D: GSSG

        # Load diffusion coefficients from pre-computed 'ab'
        ab8_im1 = ab[8, i-1]
        ab8_i   = ab[8, i]
        ab8_ip1 = ab[8, i+1]
        ab8_ip2 = ab[8, i+2]

        ab0_ip7 = ab[0, i+7]
        ab0_ip8 = ab[0, i+8]
        ab0_ip9 = ab[0, i+9]
        ab0_ip10 = ab[0, i+10]

        Fx[i+3] = ab8_im1*x0[i-1] + ab0_ip7 *x0[i+7] + (-ab8_im1-ab0_ip7 +1)*x3 + K0_star*x3 - Km0_star*x4*CH_bulk                                        - b[i+3]
        Fx[i+4] = ab8_i  *x0[i]   + ab0_ip8 *x0[i+8] + (-ab8_i  -ab0_ip8 +1)*x4 - K0_star*x3 + Km0_star*x4*CH_bulk                                        - b[i+4]
        Fx[i+5] = ab8_ip1*x0[i+1] + ab0_ip9 *x0[i+9] + (-ab8_ip1-ab0_ip9 +1)*x5                                    + K2_star*(x5**2)     - Km2_star*x6    - b[i+5]
        Fx[i+6] = ab8_ip2*x0[i+2] + ab0_ip10*x0[i+10]+ (-ab8_ip2-ab0_ip10+1)*x6                                    - (K2_star*(x5**2))/2 + Km2_star*x6/2  - b[i+6]

        i += 4

    # --- Surface Boundary Residuals ---
    Fx[0] = x0[4] - x0[0]                       # A: Zero Flux
    Fx[1] = x0[1]*ab[4,1] + x0[2]*ab[3,2] - x0[5] # B: BV flux
    Fx[2] = x0[1]*ab[5,1] + x0[2]*ab[4,2] - x0[6] # C: BV flux
    Fx[3] = x0[7] - x0[3]                       # D: Zero Flux

    # --- Bulk Boundary Residuals ---
    Fx[4*n-4] = x0[4*n-4] - 1           # A matches Bulk
    Fx[4*n-3] = x0[4*n-3] - CB_bulk     # B matches Bulk
    Fx[4*n-2] = x0[4*n-2]               # C matches Bulk (0)
    Fx[4*n-1] = x0[4*n-1]               # D matches Bulk (0)

    return Fx

def newton_raphson_solver(n, X, x0, delta_T, dB, dC, dD, K0_star, Km0_star, CH_bulk, K2_star, Km2_star, alpha, Theta, K1, h, DC_div_DB, b, Fx, ab, tol=1e-9, max_iter=100):
    """
    Solves the nonlinear system using Newton-Raphson iteration.
    Updates x0 in-place.
    """
    for _ in range(max_iter):
        ab = compute_jacobian(n, X, x0, ab, delta_T, dB, dC, dD, K0_star, Km0_star, CH_bulk, K2_star, Km2_star, alpha, Theta, K1, h, DC_div_DB)
        Fx = compute_Fx(n, x0, ab, Fx, b, K0_star, Km0_star, CH_bulk, K2_star, Km2_star) 
        dx = la.solve_banded((4, 4), ab, -Fx)
        x0 += dx

    return x0

# =============================================================================
# 9. SIMULATION LOOP & EXPORT
# =============================================================================

flux = []
pot = []

# Initialize State Vector (Interleaved: A, B, C, D at each node)
x0 = np.zeros(4*n)
x0[::4] = CA_bulk   # Species A
x0[1::4] = CB_bulk  # Species B
# Species C and D init at 0

b = np.copy(x0)
Theta = theta_i
    
# Main Time Loop
for k in range(m):
    # Update Potential
    if k < m/2:
            Theta += delta_theta
    else:
            Theta -= delta_theta

    # Solve for new concentrations
    x_solution = newton_raphson_solver(n, X, x0, delta_T, dB, dC, dD, K0_star, Km0_star, CH_bulk, K2_star, Km2_star, alpha, Theta, K1, h, DC_div_DB, b, Fx, ab)
    
    # Update History
    f_BV = math.exp(-alpha*Theta)
    b = np.copy(x_solution)
    x0 = np.copy(x_solution)
    
    # Calculate Flux (Current)
    flux_B_C = f_BV*K1*(x_solution[1]*math.exp(Theta)-x_solution[2])
    flux.append(flux_B_C)
    pot.append(Theta)

# Dimensionalize Results
I = [i*n_e*F*A*DB*cA_bulk*1E6/epsilon for i in flux]
V = [i*R*T/F + Ef - 0.235 for i in pot] # vs Ag/AgCl

# Save to file
data_1 = np.column_stack((V, I))
np.savetxt('1_V_2_I_CEC2.txt', data_1, delimiter= ',')

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")