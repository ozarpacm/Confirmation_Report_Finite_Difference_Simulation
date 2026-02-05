# =============================================================================
#   CYCLIC VOLTAMMETRY SIMULATION: CEC2E MECHANISM (RSD PATHWAY)
# =============================================================================
#
#   Description:
#       Simulation of a complex multi-step electrochemical mechanism involving
#       a pre-equilibrium proton transfer, a primary electron transfer, a
#       radical-substrate dimerization (RSD), and a secondary electron transfer.
#
#   Mechanism Modeled:
#       1. (C) Chemical: Protolysis Equilibrium
#       2. (E) Electrochemical: Primary Oxidation
#       3. (C) Chemical: Radical-Substrate Dimerization
#       4. (E) Electrochemical: Secondary Oxidation of the Dimer
#
# =============================================================================
#   REACTION SCHEME & KINETIC MODEL
# =============================================================================
#
#   0. Primary Electrode Reaction:
#      (E0)   A - e- <-->  B
#
#   1. Chemical Coupling (RSD):
#      (C1)   A + B  <-->  C
#
#   2. Secondary Electrode Reaction:
#      (E2)   C - e- <-->  D
#
#   3. Pre-equilibrium (Protolysis):
#      (C3)   E      <-->  A + H+
#
# =============================================================================
#   VARIABLE MAPPING & NUMERICAL IMPLEMENTATION
# =============================================================================
#
#   Species Legend (Inferred from context):
#   ---------------------------------------
#   E : Protio-Species (e.g., GSH)
#   A : Deprotonated Substrate (e.g., GS-)
#   B : Primary Radical (e.g., GS*)
#   C : Radical-Substrate Adduct (e.g., GSSG*-)
#   D : Final Product (e.g., GSSG)
#   H : Proton
#
#   Reaction Index & Equations:
#   ---------------------------
#   [1]  A - e- <-->  B            (Oxidation 1: E0)
#   [2]  A + B  <-->  C            (Radical-Substrate Dimerization)
#   [3]  C - e- <-->  D            (Oxidation 2: E2)
#   [4]  E <-->  A + H             (Protolysis)
#
# =============================================================================

import numpy as np
import math
import scipy.linalg as la
from numba import njit
import time

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

Ef0 = +0.653                        # Formal potential 1 (A/B couple)
Ef2 = -1.224                        # Formal potential 2 (C/D couple)

# Electrode and Scan Settings
scan_rate = 0.1                     # Scan rate (V s^-1)
alpha0 = 0.75                       # Charge transfer coefficient (Step 1)
alpha2 = 0.5                        # Charge transfer coefficient (Step 2)
epsilon = 1.5E-3                    # Electrode radius (m)
A = (math.pi)*(epsilon**2)          # Electrode area (m^2)

# Solution Conditions
pH = 7                              # Solution pH
pKa = 9                             # pKa of species E
cE_i = 1                            # Initial total concentration (mol m^-3)
HH_constant = 10**(pH-pKa)          # Henderson-Hasselbalch constant

# =============================================================================
# 2. BULK CONCENTRATIONS (mol m^-3)
# =============================================================================

# Calculate equilibrated species based on pH
cA_bulk = cE_i*HH_constant/(HH_constant+1)  # Deprotonated Substrate (A)
cB_bulk = 0                                 # Primary Radical (B)
cC_bulk = 0                                 # Dimer/Adduct (C)
cD_bulk = 0                                 # Final Product (D)
cE_bulk = cE_i-cA_bulk                      # Protio-Species (E)
cH_bulk = (10**(-pH))*1000                  # Protons (H+)

# =============================================================================
# 3. KINETIC RATE CONSTANTS
# =============================================================================

# Reaction E0: A - e- <--> B
k0 = 5.5E-6                         # Heterogeneous rate constant (m s^-1)

# Reaction C1: A + B <--> C (Radical-Substrate Dimerization)
k1 = 6.25E5                         # Forward rate constant (m^3 mol^-1 s^-1)
Keq1 = 2000                         # Equilibrium constant
K1_SS = Keq1/SS                     # Standard state corrected Keq
km1 = k1/K1_SS                      # Reverse rate constant (s^-1)

# Reaction E2: C - e- <--> D
k2 = 1                              # Heterogeneous rate constant (m s^-1)

# Reaction C3: E <--> A + H (Protolysis)
km3 = 1E7                           # Reverse rate constant (m^3 mol^-1 s^-1)
Keq3 = (10**(-pKa))                 # Equilibrium constant
K3_SS = Keq3*SS                     # Standard state corrected Keq
k3 = K3_SS*km3                      # Forward rate constant (s^-1)

# =============================================================================
# 4. DIFFUSION COEFFICIENTS (m^2 s^-1)
# =============================================================================

DA = 4.7E-10
DB = 4.7E-10
DC = 4.5E-10
DD = 4.5E-10
DE = 4.7E-10

# =============================================================================
# 5. DIMENSIONLESS PARAMETERS (Normalization)
# =============================================================================

# Potential Normalization
theta_i = (F/(R*T))*(Ei-Ef0)        # Dimensionless initial potential
theta_v = (F/(R*T))*(Ev-Ef0)        # Dimensionless vertex potential
delta_V = 0.1E-2                    # Potential step size (V)
delta_theta = (F/(R*T))*delta_V     # Dimensionless potential step
sigma = (epsilon**2)*F*scan_rate/(DA*R*T) # Dimensionless scan rate

# Dimensionless Kinetic Parameters
K0 = k0*epsilon/DA

K1  = (epsilon**2)*cA_bulk *k1/DA
Km1 = (epsilon**2)        *km1/DA

K2 = k2*epsilon/DC

K3 = (epsilon**2)          *k3/DA
Km3 = (epsilon**2)*cA_bulk*km3/DA

# Dimensionless Diffusion Coefficients
dA = DA/DA
dB = DB/DA
dC = DC/DA
dD = DD/DA
dE = DE/DA
DD_div_DC = DD/DC

# Dimensionless Bulk Concentrations
CA_bulk = cA_bulk/cA_bulk
CB_bulk = cB_bulk/cA_bulk
CC_bulk = cC_bulk/cA_bulk
CD_bulk = cD_bulk/cA_bulk
CE_bulk = cE_bulk/cA_bulk
CH_bulk = cH_bulk/cA_bulk

# =============================================================================
# 6. SPATIAL AND TEMPORAL GRID GENERATION
# =============================================================================

delta_T = delta_theta/sigma         # Dimensionless time step
maxT = 2*abs(theta_v-theta_i)/sigma # Total simulation time
d_max = max(dA, dB, dC, dD, dE)     # Max diffusion coefficient
maxX = 6*math.sqrt(d_max*maxT)      # Outer boundary
omega = 1.015                       # Grid expansion factor
h = 1e-9                            # Initial spatial step size

X = [0.0, h, 2*h]                   # Initialize Grid
x = 2*h
c = 1
while x < maxX:                     # Generate Grid
    x += h*(omega**c)
    X.append(x)
    c += 1

# Pre-calculated kinetic-temporal terms
K1_star  = K1*delta_T
Km1_star = Km1*delta_T

K3_star = K3*delta_T
Km3_star = Km3*delta_T

n = len(X)
m = round(maxT/delta_T)

# Initialize solver matrices
ab = np.zeros((11,5*n))
Fx = np.zeros(5*n)

# =============================================================================
# 7. SOLVER FUNCTIONS (JACOBIAN & RESIDUAL)
# =============================================================================

@njit
def compute_jacobian(x0, X, ab, n, delta_T, dB, dC, dD, dE, K0, K2,
                     K1_star, Km1_star, K3_star, Km3_star,
                     CH_bulk, alpha0, alpha2, Theta0, Theta2,
                     h, DD_div_DC):
    """
    Constructs the Jacobian matrix for the implicit solver.
    System involves 5 coupled species: A, B, C, D, E.
    """
    i = 1

    # Loop over internal spatial nodes
    for a in range(1, n-1):
        delX_p = X[a+1] - X[a]
        delX_m = X[a] - X[a-1]

        # Diffusion coefficients (Inverse h terms)
        denom_common_m = delX_m*delX_p + delX_m**2
        inv_hm = 2*delta_T/denom_common_m
        inv_hm_dB = dB*inv_hm; inv_hm_dC = dC*inv_hm
        inv_hm_dD = dD*inv_hm; inv_hm_dE = dE*inv_hm

        denom_common_p = delX_m*delX_p + delX_p**2
        inv_hp = 2*delta_T/denom_common_p
        inv_hp_dB = dB*inv_hp; inv_hp_dC = dC*inv_hp
        inv_hp_dD = dD*inv_hp; inv_hp_dE = dE*inv_hp

        # Index Mapping for 5 Species (A, B, C, D, E)
        i0 = i
        i1,  i2,  i3,  i4      = i0+1,  i0+2,  i0+3,  i0+4
        i5,  i6,  i7,  i8, i9  = i0+5,  i0+6,  i0+7,  i0+8, i0+9
        i10, i11, i12, i13     = i0+10, i0+11, i0+12, i0+13
        
        # Local concentrations for linearization
        x4 = x0[i4] # Species A
        x5 = x0[i5] # Species B

        # --- Species A (Index i4) ---
        ab[10,i0-1] = -inv_hm                                       # Prev Node
        ab[0,i9]    = -inv_hp                                       # Next Node
        ab[5,i4]    = (+inv_hm+inv_hp+1) + K1_star*x5 + Km3_star*CH_bulk # Diagonal
        ab[4,i5]    = +K1_star*x4
        ab[3,i6]    = -Km1_star
        ab[1,i8]    = -K3_star

        # --- Species B (Index i5) ---
        ab[10,i0]   = -inv_hm_dB
        ab[0,i10]   = -inv_hp_dB
        ab[5,i5]    = (+inv_hm_dB+inv_hp_dB+1) + K1_star*x4         # Diagonal
        ab[6,i4]    = +K1_star*x5
        ab[4,i6]    = -Km1_star

        # --- Species C (Index i6) ---
        ab[10,i1]   = -inv_hm_dC
        ab[0,i11]   = -inv_hp_dC
        ab[5,i6]    = (+inv_hm_dC+inv_hp_dC+1) + Km1_star           # Diagonal
        ab[7,i4]    = -K1_star*x5
        ab[6,i5]    = -K1_star*x4

        # --- Species D (Index i7) ---
        ab[10,i2]   = -inv_hm_dD
        ab[0,i12]   = -inv_hp_dD
        ab[5,i7]    = (+inv_hm_dD+inv_hp_dD+1)                      # Diagonal (No kinetics)

        # --- Species E (Index i8) ---
        ab[10,i3]   = -inv_hm_dE
        ab[0,i13]   = -inv_hp_dE
        ab[5,i8]    = (+inv_hm_dE+inv_hp_dE+1) + K3_star            # Diagonal
        ab[9,i4]    = -Km3_star*CH_bulk

        i += 5

    # --- Boundary Conditions (Surface x=0) ---
    f_BV0 = math.exp(-alpha0*Theta0)
    f_BV2 = math.exp(-alpha2*Theta2)

    Kred0 = f_BV0*K0
    Kox0 = f_BV0*K0*math.exp(Theta0)
    Kred2 = f_BV2*K2
    Kox2 = f_BV2*K2*math.exp(Theta2)

    # Couple 1 (A/B)
    ab[5, 0] = 1+h*Kox0
    ab[4, 1] = -h*Kred0
    ab[0, 5] = -1

    ab[6, 0] = -(1/dB)*h*Kox0
    ab[5, 1] = 1+(1/dB)*h*Kred0
    ab[0, 6] = -1

    # Couple 2 (C/D)
    ab[5, 2] = 1+h*Kox2
    ab[4, 3] = -h*Kred2
    ab[0, 7] = -1

    ab[6, 2] = -(1/DD_div_DC)*h*Kox2
    ab[5, 3] = 1+(1/DD_div_DC)*h*Kred2
    ab[0, 8] = -1

    # Species E (Zero Flux)
    ab[5, 4]  = -1
    ab[0, 9]  = 1

    # --- Boundary Conditions (Bulk x=Inf) ---
    ab[5, 5*n-1] = 1
    ab[5, 5*n-2] = 1
    ab[5, 5*n-3] = 1
    ab[5, 5*n-4] = 1
    ab[5, 5*n-5] = 1

    return ab

@njit
def compute_Fx(x0, ab, Fx, b, n, K1_star, Km1_star, K3_star, Km3_star,
               CH_bulk, CA_bulk, CC_bulk, CE_bulk):
    """
    Computes the Residual Vector F(x) for the 5-species system.
    """
    i = 1
    for a in range(1, n-1):
        # Conservation Equations (Diff + Kinetic - C_old)
        Fx[i+4]  = ab[10,i-1]*x0[i-1] + ab[0,i+9] *x0[i+9]  + (-ab[10,i-1]-ab[0,i+9] +1)*x0[i+4] + K1_star*x0[i+4]*x0[i+5] - Km1_star*x0[i+6] - K3_star*x0[i+8] + Km3_star*x0[i+4]*CH_bulk - b[i+4]
        Fx[i+5]  = ab[10,i]  *x0[i]   + ab[0,i+10]*x0[i+10] + (-ab[10,i]  -ab[0,i+10]+1)*x0[i+5] + K1_star*x0[i+4]*x0[i+5] - Km1_star*x0[i+6]                                              - b[i+5]
        Fx[i+6]  = ab[10,i+1]*x0[i+1] + ab[0,i+11]*x0[i+11] + (-ab[10,i+1]-ab[0,i+11]+1)*x0[i+6] - K1_star*x0[i+4]*x0[i+5] + Km1_star*x0[i+6]                                              - b[i+6]
        Fx[i+7]  = ab[10,i+2]*x0[i+2] + ab[0,i+12]*x0[i+12] + (-ab[10,i+2]-ab[0,i+12]+1)*x0[i+7]                                                                                           - b[i+7]
        Fx[i+8]  = ab[10,i+3]*x0[i+3] + ab[0,i+13]*x0[i+13] + (-ab[10,i+3]-ab[0,i+13]+1)*x0[i+8]                                              + K3_star*x0[i+8] - Km3_star*x0[i+4]*CH_bulk - b[i+8]

        i += 5

    # Surface Residuals (Butler-Volmer & Flux)
    Fx[0] = x0[0]*ab[5,0] + x0[1]*ab[4,1] - x0[5]   # A/B Flux Match
    Fx[1] = x0[0]*ab[6,0] + x0[1]*ab[5,1] - x0[6]   # B/A Flux Match
    Fx[2] = x0[2]*ab[5,2] + x0[3]*ab[4,3] - x0[7]   # C/D Flux Match
    Fx[3] = x0[2]*ab[6,2] + x0[3]*ab[5,3] - x0[8]   # D/C Flux Match
    Fx[4] = x0[9] - x0[4]                           # E Zero Flux

    # Bulk Residuals (Dirichlet)
    Fx[5*n-5] = x0[5*n-5] - CA_bulk
    Fx[5*n-4] = x0[5*n-4] 
    Fx[5*n-3] = x0[5*n-3] - CC_bulk
    Fx[5*n-2] = x0[5*n-2]
    Fx[5*n-1] = x0[5*n-1] - CE_bulk

    return Fx

def newton_raphson_solver(x0, X, ab, Fx, b, n, delta_T, dB, dC, dD, dE, K0, K2, K1_star, Km1_star, K3_star, Km3_star, CH_bulk, CA_bulk, CC_bulk, CE_bulk, alpha0, alpha2, Theta0, Theta2, h, DD_div_DC, tol=1e-9, max_iter=100):
    """
    Newton-Raphson Solver for the nonlinear system.
    """
    for _ in range(max_iter):

        ab = compute_jacobian(x0, X, ab, n, delta_T, dB, dC, dD, dE, K0, K2,
                     K1_star, Km1_star, K3_star, Km3_star,
                     CH_bulk, alpha0, alpha2, Theta0, Theta2,
                     h, DD_div_DC)
        Fx = compute_Fx(x0, ab, Fx, b, n, K1_star, Km1_star, K3_star, Km3_star,
               CH_bulk, CA_bulk, CC_bulk, CE_bulk)
        dx = la.solve_banded((5, 5), ab, -Fx)
        x0 += dx
        if np.linalg.norm(dx, np.inf) < tol:
            return x0
    
    return x0

# =============================================================================
# 8. SIMULATION INITIALIZATION & LOOP
# =============================================================================

flux = []
pot = []

# Initialize State Vector
x0 = np.zeros(5*n)
x0[::5] = CA_bulk
x0[2::5] = CC_bulk
x0[4::5] = CE_bulk
b = np.copy(x0)

E = Ei

# Main Time Loop
for k in range(m):
    # Scan Potential
    if k < m/2:
        E += delta_V  
    else:
        E -= delta_V
    
    # Calculate Thetas for both couples
    Theta0 = (F/(R*T))*(E-Ef0)
    Theta2 = (F/(R*T))*(E-Ef2)
    
    # Solve
    x_solution = newton_raphson_solver(x0, X, ab, Fx, b, n, delta_T, dB, dC, dD, dE, K0, K2, K1_star, Km1_star, K3_star, Km3_star, CH_bulk, CA_bulk, CC_bulk, CE_bulk, alpha0, alpha2, Theta0, Theta2, h, DD_div_DC)
    
    # Store History
    f_BV0 = math.exp(-alpha0*Theta0)
    f_BV2 = math.exp(-alpha2*Theta2)
    
    # Total Current = Flux(A->B) + Flux(C->D)
    flux_value = (DA*f_BV0*K0*(math.exp(Theta0)*x_solution[0]-x_solution[1]) + DC*f_BV2*K2*(math.exp(Theta2)*x_solution[2]-x_solution[3]))
    flux.append(float(flux_value))
    pot.append(float(Theta0))
    b = np.copy(x_solution)
    x0 = np.copy(x_solution)

# =============================================================================
# 9. DATA POST-PROCESSING AND EXPORT
# =============================================================================

I = [i*n_e*F*(math.pi)*epsilon*cA_bulk*1E6 for i in flux]
V = [i*R*T/F + Ef0 - 0.235 for i in pot]

data_1= list(zip(V, I))
np.savetxt('1_V_2_I_CEC2E.txt', data_1, delimiter= ',')

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")