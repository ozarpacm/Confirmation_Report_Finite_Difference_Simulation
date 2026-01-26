# =============================================================================
#   CYCLIC VOLTAMMETRY SIMULATION: Fe3+ MEDIATED GSH OXIDATION
# =============================================================================
#
#   Description:
#       Simulation of the electrochemical oxidation of Glutathione (GSH) 
#       mediated by electrochemically generated Fe3+ (ferro-ferricyanide redox).
#       The model assumes Butler-Volmer kinetics at a macroelectrode.
#
#   Mechanism Modeled:
#       Proton-Coupled Electron Transfer (PCET) via the stepwise 
#       Proton Transfer followed by Electron Transfer (PET) pathway.
#
#   Abbreviations:
#       CV   : Cyclic Voltammetry
#       PCET : Proton-Coupled Electron Transfer
#       PT   : Proton Transfer
#       ET   : Electron Transfer
#       PET  : Proton Transfer followed by Electron Transfer
#       RRD  : Radical-Radical Dimerization
#       RSD  : Radical-Substrate Dimerization
#
# =============================================================================
#   REACTION SCHEME & KINETIC MODEL
# =============================================================================
#
#   1. PCET Steps:
#      (PT)   GSH             <-->  GS- + H+
#      (Elec) Fe2+ - e-       <-->  Fe3+
#      (ET)   Fe3+ + GS-      <-->  Fe2+ + GS*
#
#   2. Chemical Steps:
#      (RsD)  GS* + GS-       <-->  GSSG*-
#             Fe3+ + GSSG*-   <-->  Fe2+ + GSSG
#      (RRD)  2GS* <-->  GSSG
#
# =============================================================================
#   VARIABLE MAPPING & NUMERICAL IMPLEMENTATION
# =============================================================================
#   To improve code readability and generality, chemical species are mapped 
#   to abstract variables (A, B, X, Y...). Rate/Equilibrium constants follow 
#   the reaction index (e.g., k0, Keq0 correspond to Reaction 0).
#
#   Species Legend:
#   ---------------
#   B : Fe2+        A : Fe3+   
#   X : GSH         Y : GS-         Z : GS*  
#   M : GSSG*-      N : GSSG
#   H : H+

#   Reaction Index & Equations:
#   ---------------------------
#   [0]  X      <-->  Y + H        (GSH deprotonation)
#   [1]  B - e- <-->  A            (Electrode Reaction)
#   [2]  A + Y  <-->  B + Z        (Fe3+ oxidizes GS-)
#   [3]  Y + Z  <-->  M            (Radical-Substrate dimerization)
#   [4]  A + M  <-->  N + B        (Oxidation of intermediate GSSG*-)
#   [5]  2Z     <-->  N            (Radical dimerization)
#
# =============================================================================

import numpy as np
import math
import scipy.linalg as la
from numba import njit

# =============================================================================
# 1. PHYSICAL CONSTANTS AND SIMULATION PARAMETERS
# =============================================================================

R = 8.314                           # Gas constant (J mol^-1 K^-1)
T = 298                             # Temperature (K)
F = 96485                           # Faraday constant (C mol^-1)
n_e = 1                             # Number of electrons transferred
SS = 1000                           # Standard state correction factor (1 mol m^-3 = 1000 M)

# Potential Parameters
Ei = +0.03603                       # Initial potential (V)
Ev = +0.93538                       # Vertex potential (V)
Ef = 0.43                           # Formal potential (V)

# Electrode and Scan Settings
scan_rate = 0.1                     # Scan rate (V s^-1)
alpha = 0.5                         # Charge transfer coefficient
epsilon = 1.5E-3                    # Electrode radius (m)
A = (math.pi)*(epsilon**2)          # Electrode area (m^2)

# Solution Conditions
pH   = 7                            # Solution pH
pKa  = 9.0                          # Apparent pKa of the thiol group
cX_i = 1                            # Initial total concentration of GSH (mol m^-3)

# Henderson-Hasselbalch Calculation
HH_constant = 10**(pH-pKa)          # Henderson-Hasselbach constant

# =============================================================================
# 2. BULK CONCENTRATIONS (mol m^-3)
# =============================================================================

cA_bulk = 0                         # Fe3+
cB_bulk = 1                         # Fe2+
cZ_bulk = 0                         # GS*
cM_bulk = 0                         # GSSG*-
cN_bulk = 0                         # GSSG
cH_bulk = (10**(-pH))*1000          # H+

# Calculate equilibrated GSH species based on pH
cY_bulk = (cX_i * HH_constant/      # GS-
          (HH_constant + 1))
cX_bulk = cX_i-cY_bulk              # GSH

# =============================================================================
# 3. KINETIC RATE CONSTANTS
# =============================================================================
# Reaction 0: Acid/Base Equilibrium
km0 = 1E7                           # Reverse rate constant (m^3 mol^-1 s^-1)
Keq0 = (10**(-pKa))                 # Equilibrium constant
K0_SS = Keq0*SS                     # Standard state corrected Keq
k0 = K0_SS*km0                      # Forward rate constant (s^-1)

# Reaction 1: Electrochemical oxidation of Fe2+
k1 = 2.40E-5                        # Standard heterogeneous rate constant (m s^-1)

# Reaction 2: GS- oxidation
km2 = 1E7                           # Reverse rate constant (m^3 mol^-1 s^-1)
Keq2 = 2.1E-4                       # Equilibrium constant
k2 = km2*Keq2                       # Forward rate constant (m^3 mol^-1 s^-1)

# Reaction 3: RSD
k3 = 0                              # Forward rate constant (m^3 mol^-1 s^-1)
Keq3 = 338                          # Equilibrium constant
K3_SS = Keq3/SS                     # Standard state corrected Keq
km3 = k3/K3_SS                      # Reverse rate constant (m^3 mol^-1 s^-1)

# Reaction 4: GSSG*- oxidation
k4 = 0                              # Forward rate constant (m^3 mol^-1 s^-1)
Keq4 = 1                            # Equilibrium constant
km4 = k4/Keq4                       # Reverse rate constant (m^3 mol^-1 s^-1)

# Reaction 5: RRD
k5 = (1E7)/2                        # Forward rate constant (m^3 mol^-1 s^-1)
Keq5 = 3.4E34                       # Equilibrium constant
K5_SS = Keq5/SS                     # Standard state corrected Keq
km5 = k5/K5_SS                      # Reverse rate constant (s^-1)

# =============================================================================
# 4. DIFFUSION COEFFICIENTS (m^2 s^-1)
# =============================================================================

DA = 7.62E-10                       # Fe3+
DB = 6.50E-10                       # Fe2+
DX = 5.7E-10                        # GSH
DY = 5.7E-10                        # GS-
DZ = 5.7E-10                        # GS*
DM = 4.5E-10                        # GSSG*-
DN = 4.5E-10                        # GSSG

# =============================================================================
# 5. DIMENSIONLESS PARAMETERS (Normalization)
# =============================================================================
# Potential Normalization
theta_i = (F/(R*T))*(Ei-Ef)         # Dimensionless initial potential
theta_v = (F/(R*T))*(Ev-Ef)         # Dimensionless vertex potential
delta_V = 0.5E-3                    # Potential step size (V)
delta_theta = (F/(R*T))*delta_V     # Dimensionless potential step
sigma = ((epsilon**2)*F*            # Dimensionless scan rate
        scan_rate/(DA*R*T))

# Dimensionless Kinetic Parameters
K0  = (epsilon**2)        *k0/DA
Km0 = (epsilon**2)*cB_bulk*km0/DA

K1 = k1*epsilon/DB

K2  = (epsilon**2)*cB_bulk*k2/DA
Km2 = (epsilon**2)*cB_bulk*km2/DA

K3  = (epsilon**2)*cB_bulk*k3/DA
Km3 = (epsilon**2)        *km3/DA

K4  = (epsilon**2)*cB_bulk*k4/DA
Km4 = (epsilon**2)*cB_bulk*km4/DA

K5  = (epsilon**2)*cB_bulk*k5/DA
Km5 = (epsilon**2)        *km5/DA

# Dimensionless Diffusion Coefficients (D_species / D_A)
dA = DA/DA
dB = DB/DA
dX = DX/DA
dY = DY/DA
dZ = DZ/DA
dM = DM/DA
dN = DN/DA

# Dimensionless Bulk Concentrations (C_species / C_ref)
CA_bulk = cA_bulk/cB_bulk
CB_bulk = cB_bulk/cB_bulk
CX_bulk = cX_bulk/cB_bulk
CY_bulk = cY_bulk/cB_bulk
CZ_bulk = cZ_bulk/cB_bulk
CM_bulk = cM_bulk/cB_bulk
CN_bulk = cN_bulk/cB_bulk
CH_bulk = cH_bulk/cB_bulk

# =============================================================================
# 6. SPATIAL AND TEMPORAL GRID GENERATION
# =============================================================================
# Temporal Grid
delta_T = delta_theta/sigma         # Dimensionless time step
maxT = 2*abs(theta_v-theta_i)/sigma # Total simulation time (dimensionless)

# Spatial Grid Boundaries
d_max = (max(dA, dB, dX, dY, dZ,    # Max diffusion coefficient
              dM, dN))
maxX = 6*math.sqrt(d_max*maxT)      # Outer simulation boundary (6 * diffusion length)

# Exponential Grid Setup
omega = 1.02                        # Grid expansion factor
h = 1e-8                            # Initial spatial step size

# Initialize Grid Points (X)
# The first three points are fixed to facilitate 3-point flux approximation at X=0
X = [0.0, h, 2*h]
x = 2*h                             # Current position
c = 1                               # Expansion counter

# Generate remaining grid points
while x < maxX:
    x += h*(omega**c)
    X.append(x)
    c += 1

# =============================================================================
# 7. PRE-CALCULATION OF KINETIC-TEMPORAL TERMS
# =============================================================================
# Pre-multiplying the dimensionless rate constants by the time step (delta_T)
# simplifies the kinetic updates in the main solver loop. These "star" terms 
# represent the dimensionless reaction extent per time step

K0_star  = K0*delta_T
Km0_star = Km0*delta_T

K2_star  = K2*delta_T
Km2_star = Km2*delta_T

K3_star  = K3*delta_T
Km3_star = Km3*delta_T

K4_star  = K4*delta_T
Km4_star = Km4*delta_T

K5_star  = K5*delta_T
Km5_star = Km5*delta_T

# =============================================================================
# 9. SOLVER INITIALIZATION & JACOBIAN CONSTRUCTION
# =============================================================================

n = len(X)                  # Number of spatial grid points
m = round(maxT/delta_T)     # Total number of time steps

# Initialize the sparse matrix 'ab' for the band linear solver (scipy.linalg.solve_banded).
# The shape (15, 7*n) indicates a banded matrix structure:
#   - Rows: 15 diagonals (7 upper, 7 lower, 1 main) required for 7 coupled species.
#   - Cols: 7*n represents the flattened state vector (7 species at n nodes).
ab = np.zeros((15,7*n))


@njit
def compute_jacobian(x0, X, ab, delta_T, CH_bulk, Theta):
    """
    Constructs the Jacobian matrix for the implicit finite difference solver.
    
    The system solves J * dC = -F, where J is the Jacobian of the residual vector.
    This function populates 'ab' with partial derivatives of the discretized PDEs.
    
    Parameters:
    x0      : Array of current concentrations (flattened)
    X       : Array of spatial grid positions
    ab      : Banded matrix to populate
    Theta   : Dimensionless potential (theta) for Butler-Volmer BCs
    """
    i = 1  # Index counter for the flattened state vector
    
    # Loop over internal spatial nodes (excluding boundaries 0 and n-1)
    for a in range(1, n-1):
        # The grid expands exponentially, so delX_p != delX_m.
        # We calculate the coefficients for the discrete diffusion operators (alpha, beta, gamma)
        # using a centeral difference scheme adapted for non-uniform spacing.

        # Grid steps
        delX_p = X[a+1] - X[a]   # Step size to the right (Node a -> a+1)
        delX_m = X[a] - X[a-1]   # Step size to the left  (Node a -> a-1)

        # Pre-calculate diffusion factors (Inverse h terms)
        # 'inv_hm' corresponds to the coefficient for (alpha)
        # 'inv_hp' corresponds to the coefficient for (gamma)
        # Note: These include the 2*delta_T factor for the Implicit time-stepping.
        denom_common_m = delX_m**2 + delX_m*delX_p
        inv_hm = 2*delta_T/denom_common_m           # alpha
        denom_common_p = delX_p**2 + delX_m*delX_p
        inv_hp = 2 * delta_T/denom_common_p         # gamma

        # Scale geometric factors by species-specific diffusion coefficients
        inv_hm_dB = dB * inv_hm
        inv_hm_dX = dX * inv_hm
        inv_hm_dY = dY * inv_hm
        inv_hm_dZ = dZ * inv_hm
        inv_hm_dM = dM * inv_hm
        inv_hm_dN = dN * inv_hm

        inv_hp_dB = dB * inv_hp
        inv_hp_dX = dX * inv_hp
        inv_hp_dY = dY * inv_hp
        inv_hp_dZ = dZ * inv_hp
        inv_hp_dM = dM * inv_hp
        inv_hp_dN = dN * inv_hp

        # ---------------------------------------------------------------------
        # B. Index Mapping (Sliding Window Strategy)
        # ---------------------------------------------------------------------
        # We use a sliding window of indices relative to 'i0' (which anchors the Previous Node).
        
        # Define base anchor
        i0 = i

        # --- 1. Define Indices for Spatial Neighbors ---
        # Range [i0-1 ... i5]  -> Previous Node (a-1) (Used for Left Diffusion)
        # Range [i6   ... i12] -> Current Node (a)    (Used for Reaction & Self-Diffusion)
        # Range [i13  ... i19] -> Next Node (a+1)     (Used for Right Diffusion)

        # Unroll indices for the 7 species within the current sliding window
        # (Species Order: 0:Fe3+, 1:Fe2+, 2:GSH, 3:GS-, 4:GS*, 5:GSSG*-, 6:GSSG)
        i1, i2, i3, i4          = i0+1, i0+2, i0+3, i0+4
        i5, i6, i7, i8, i9      = i0+5, i0+6, i0+7, i0+8, i0+9
        i10, i11, i12, i13, i14 = i0+10, i0+11, i0+12, i0+13, i0+14
        i15, i16, i17, i18, i19 = i0+15, i0+16, i0+17, i0+18, i0+19

        # --- 2. Load Current Concentrations (Linearization Points) ---
        # We retrieve the concentrations at the Current Node (indices i6 to i12)
        # to calculate the Jacobian elements for the non-linear kinetic terms.

        x6  = x0[i6]   # C_Fe3+  (Current Node)
        x7  = x0[i7]   # C_Fe2+  (Current Node)
        x9  = x0[i9]   # C_GS-   (Current Node)
        x10 = x0[i10]  # C_GS* (Current Node)
        x11 = x0[i11]  # C_GSSG*-(Current Node)
        x12 = x0[i12]  # C_GSSG  (Current Node)

        # ---------------------------------------------------------------------
        # C. Jacobian Matrix Population
        # ---------------------------------------------------------------------
        # 'ab' is a banded matrix with shape (15, 7*n).
        # Row mapping for scipy.linalg.solve_banded (Upper bandwidth = 7):
        #   Row 0  : Far Upper Diagonal (Coupling to i+7 / Next Node)
        #   Row 7  : Main Diagonal      (Self coupling)
        #   Row 14 : Far Lower Diagonal (Coupling to i-7 / Prev Node)
        #   Rows 1-6, 8-13: Intra-node coupling (Kinetics between species at same node)

        # --- Equations for Species 0: Fe3+ (Index i6) ---
        ab[14, i0-1] = -inv_hm                                          # alpha
        ab[0, i13]   = -inv_hp                                          # gamma
        ab[7, i6]    = (inv_hm+inv_hp+1) + K2_star*x9 + K4_star*x11     # beta + d(Eq0)/d(Fe3+)
        ab[6, i7]    = -Km2_star*x10 - Km4_star*x12                     # d(Eq0)/d(Fe2+)
        ab[4, i9]    = +K2_star*x6                                      # d(Eq0)/d(GS-)
        ab[3, i10]   = -Km2_star*x7                                     # d(Eq0)/d(GS*)
        ab[2, i11]   = +K4_star*x6                                      # d(Eq0)/d(GSSG*-)
        ab[1, i12]   = -Km4_star*x7                                     # d(Eq0)/d(GSSG)

        # --- Equations for Species 1: Fe2+ (Index i7) ---
        ab[14, i0]   = -inv_hm_dB
        ab[0, i14]   = -inv_hp_dB
        ab[7, i7]    = (inv_hm_dB+inv_hp_dB+1) + Km2_star*x10 + Km4_star*x12
        ab[8, i6]    = -K2_star*x9 - K4_star*x11
        ab[5, i9]    = -K2_star*x6
        ab[4, i10]   = +Km2_star*x7
        ab[3, i11]   = -K4_star*x6
        ab[2, i12]   = +Km4_star*x7

        # --- Equations for Species 2: GSH (Index i8) ---
        ab[14, i1]   = -inv_hm_dX
        ab[0, i15]   = -inv_hp_dX
        ab[7, i8]    = (inv_hm_dX+inv_hp_dX+1) + K0_star
        ab[6, i9]    = - Km0_star * CH_bulk

        # --- Equations for Species 3: GS- (Index i9) ---
        ab[14, i2]   = -inv_hm_dY
        ab[0, i16]   = -inv_hp_dY
        ab[7, i9]    = (inv_hm_dY+inv_hp_dY+1) + Km0_star*CH_bulk + K2_star*x6 + K3_star*x10
        ab[10, i6]   =  K2_star*x9
        ab[9, i7]    = - Km2_star*x10
        ab[8, i8]    = - K0_star
        ab[6, i10]   = - Km2_star*x7 + K3_star * x9
        ab[5, i11]   = - Km3_star

        # --- Equations for Species 4: GS* (Index i10) ---
        ab[14, i3]   = -inv_hm_dZ
        ab[0, i17]   = -inv_hp_dZ
        ab[7, i10]   = (inv_hm_dZ+inv_hp_dZ+1) + Km2_star*x7 + K3_star*x9 + 4*K5_star*x10
        ab[11, i6]   = - K2_star*x9
        ab[10, i7]   = + Km2_star*x10
        ab[8, i9]    = - K2_star*x6 + K3_star*x10
        ab[6, i11]   = - Km3_star
        ab[5, i12]   = - 2*Km5_star

        # --- Equations for Species 5: GSSG*- (Index i11) ---
        ab[14, i4]   = -inv_hm_dM
        ab[0, i18]   = -inv_hp_dM
        ab[7, i11]   = (inv_hm_dM+inv_hp_dM+1) + Km3_star + K4_star*x6
        ab[12, i6]   = + K4_star * x11
        ab[11, i7]   = - Km4_star * x12
        ab[9, i9]    = - K3_star * x10
        ab[8, i10]   = - K3_star * x9
        ab[6, i12]   = - Km4_star * x7

        # --- Equations for Species 6: GSSG (Index i12) ---
        ab[14, i5]   = -inv_hm_dN
        ab[0, i19]   = -inv_hp_dN
        ab[7, i12]   = (inv_hm_dN+inv_hp_dN+1) + Km4_star*x7 + Km5_star
        ab[13, i6]   = - K4_star * x11
        ab[12, i7]   = + Km4_star * x12
        ab[9, i10]   = - 2 * K5_star * x10
        ab[8, i11]   = - K4_star * x6

        # Step index by 7 to move to the next spatial block
        i += 7

    # -------------------------------------------------------------------------
    # D. BOUNDARY CONDITIONS (Surface x=0)
    # -------------------------------------------------------------------------
    # The boundary conditions at the electrode surface are governed by:
    # 1. Butler-Volmer Kinetics for electroactive species (Fe3+/Fe2+).
    # 2. Zero Flux (Insulation) for non-electroactive species.

    # Calculate electrochemical rate constants based on current potential (Theta)
    # f_BV0: Potential-dependent prefactor
    # Kred : Rate constant for Reduction (Fe3+ + e- -> Fe2+)
    # Kox  : Rate constant for Oxidation (Fe2+ -> Fe3+ + e-)
    f_BV0 = math.exp(-alpha*Theta)
    Kred = f_BV0*K1 
    Kox = f_BV0*K1*math.exp(Theta)

    # --- BC for Fe3+ (Species 0) ---
    # Dimensionalised: DB * dcB/dx =  k_ox*cB - k_red*cA
    ab[7, 0] = (1+h*dB*Kred)  # Coefficient of Fe2+ at the electrode surface (i = 0)
    ab[6, 1] = -h*dB*Kox      # Coefficient of Fe3+ at the electrode surface (i = 0)
    ab[0, 7] = -1             # Coefficient of Fe2+ at i = 1. 

    ab[8, 0] = -h*Kred        # Coefficient of Fe2+ at the electrode surface (i = 0)
    ab[7, 1] = 1+h*Kox        # Coefficient of Fe3+ at the electrode surface (i = 0)
    ab[0, 8] = -1             # Coefficient of Fe3+ at i = 1. 

    # --- BC for Non-Electroactive Species (Species 2-6) ---
    # Zero Flux Condition: dC/dx = 0
    # Species 2 (GSH)
    ab[7, 2] = -1; ab[0, 9]  = +1
    # Species 3 (GS-)
    ab[7, 3] = -1; ab[0, 10] = +1
    # Species 4 (GS*)
    ab[7, 4] = -1; ab[0, 11] = +1
    # Species 5 (GSSG*-)
    ab[7, 5] = -1; ab[0, 12] = +1
    # Species 6 (GSSG)
    ab[7, 6] = -1; ab[0, 13] = +1

    # -------------------------------------------------------------------------
    # E. BOUNDARY CONDITIONS (Bulk x=Inf)
    # -------------------------------------------------------------------------
    # Dirichlet Boundary Condition: Concentrations at infinity match bulk values.
    # Implemented as an identity equation: 1 * C_last = C_bulk
    # (The value C_bulk is handled in the RHS residual vector, here we set the matrix diagonal to 1)

    ab[7, 7*n-1] = 1  # Species 6 (GSSG)
    ab[7, 7*n-2] = 1  # Species 5 (GSSG*-)
    ab[7, 7*n-3] = 1  # Species 4 (GS*)
    ab[7, 7*n-4] = 1  # Species 3 (GS-)
    ab[7, 7*n-5] = 1  # Species 2 (GSH)
    ab[7, 7*n-6] = 1  # Species 1 (Fe2+)
    ab[7, 7*n-7] = 1  # Species 0 (Fe3+)

    return ab

@njit
def compute_Fx(x0, Fx, ab, b, CH_bulk, CX_bulk, CY_bulk):

    i = 1               # Index pointer for the current spatial node (starting at Node 1)

    # Loop over internal spatial nodes (excluding boundaries 0 and n-1)
    for a in range(1,n-1):
        # The indices i-1, i, i+1 refer to spatial nodes (Left, Center, Right)
        # The offsets +0 to +6 refer to the species (Fe3+, Fe2+, GSH, GS-, GS*, GSSG*-, GSSG)
        
        # ---------------------------------------------------------------------
        # CONSERVATION EQUATIONS (Diffusion + Kinetics - C_old)
        # ---------------------------------------------------------------------
        # Note: 'ab' contains the discretized diffusion coefficients (alpha, beta, gamma).

        # --- Species 0: Fe3+ (Index i+6) ---
        Fx[i+6]  = ab[14, i-1]*x0[i-1] + ab[0, i+13]*x0[i+13] + (-ab[14, i-1]-ab[0, i+13]+1) *x0[i+6]  + K2_star*x0[i+6]*x0[i+9] - Km2_star*x0[i+7]*x0[i+10] + K4_star*x0[i+6]*x0[i+11] - Km4_star*x0[i+12]*x0[i+7] - b[i+6]
        # --- Species 1: Fe2+ (Index i+7) ---
        Fx[i+7]  = ab[14, i]  *x0[i]   + ab[0, i+14]*x0[i+14] + (-ab[14, i]  -ab[0, i+14]+1) *x0[i+7]  - K2_star*x0[i+6]*x0[i+9] + Km2_star*x0[i+7]*x0[i+10] - K4_star*x0[i+6]*x0[i+11] + Km4_star*x0[i+12]*x0[i+7] - b[i+7]
        # --- Species 2: GSH (Index i+8) ---
        Fx[i+8]  = ab[14, i+1]*x0[i+1] + ab[0, i+15]*x0[i+15] + (-ab[14, i+1]-ab[0, i+15]+1) *x0[i+8]  + K0_star*x0[i+8] - Km0_star*x0[i+9]*CH_bulk - b[i+8]
        # --- Species 3: GS- (Index i+9) ---
        Fx[i+9]  = ab[14, i+2]*x0[i+2] + ab[0, i+16]*x0[i+16] + (-ab[14, i+2]-ab[0, i+16]+1) *x0[i+9]  - K0_star*x0[i+8] + Km0_star*x0[i+9]*CH_bulk + K2_star*x0[i+6]*x0[i+9] - Km2_star*x0[i+7]*x0[i+10] + K3_star*x0[i+9]*x0[i+10] - Km3_star*x0[i+11] - b[i+9]
        # --- Species 4: GS* (Index i+10) ---
        Fx[i+10] = ab[14, i+3]*x0[i+3] + ab[0, i+17]*x0[i+17] + (-ab[14, i+3]-ab[0, i+17]+1) *x0[i+10] - K2_star*x0[i+6]*x0[i+9] + Km2_star*x0[i+7]*x0[i+10] + K3_star*x0[i+9]*x0[i+10] - Km3_star*x0[i+11] + 2*K5_star*(x0[i+10])**2 - 2*Km5_star*x0[i+12] - b[i+10]
        # --- Species 5: GSSG*- (Index i+11) ---
        Fx[i+11] = ab[14, i+4]*x0[i+4] + ab[0, i+18]*x0[i+18] + (-ab[14, i+4]-ab[0, i+18]+1) *x0[i+11] - K3_star*x0[i+9]*x0[i+10] + Km3_star*x0[i+11] + K4_star*x0[i+6]*x0[i+11] - Km4_star*x0[i+12]*x0[i+7] - b[i+11]
        # --- Species 6: GSSG (Index i+12) ---
        Fx[i+12] = ab[14, i+5]*x0[i+5] + ab[0, i+19]*x0[i+19] + (-ab[14, i+5]-ab[0, i+19]+1) *x0[i+12] - K4_star*x0[i+6]*x0[i+11] + Km4_star*x0[i+12]*x0[i+7] - K5_star*(x0[i+10])**2 + Km5_star*x0[i+12]    - b[i+12]

        i += 7 # Advance to the next spatial block


    # -------------------------------------------------------------------------
    # BOUNDARY CONDITIONS (Surface x=0)
    # -------------------------------------------------------------------------
    # Electroactive Species (Fe3+/Fe2+): Butler-Volmer Flux Balance
    Fx[0] = x0[0]*ab[7,0] + x0[1]*ab[6,1] - x0[7]
    Fx[1] = x0[0]*ab[8,0] + x0[1]*ab[7,1] - x0[8]
    
    # Non-Electroactive Species: Zero Flux
    Fx[2] = x0[9] - x0[2]
    Fx[3] = x0[10] - x0[3]
    Fx[4] = x0[11] - x0[4]
    Fx[5] = x0[12] - x0[5]
    Fx[6] = x0[13] - x0[6]

    # -------------------------------------------------------------------------
    # BOUNDARY CONDITIONS (Bulk x=Inf)
    # -------------------------------------------------------------------------
    # Dirichlet Condition: C_last_node - C_bulk = 0
    # Ensures concentrations match the bulk solution at the simulation boundary.
    Fx[7*n-7] = x0[7*n-7]             # Fe3+ (Bulk = 0)
    Fx[7*n-6] = x0[7*n-6] - 1         # Fe2+ (Bulk = 1 normalized)
    Fx[7*n-5] = x0[7*n-5] - CX_bulk   # GSH
    Fx[7*n-4] = x0[7*n-4] - CY_bulk   # GS-
    Fx[7*n-3] = x0[7*n-3]             # GS*
    Fx[7*n-2] = x0[7*n-2]             # GSSG*-
    Fx[7*n-1] = x0[7*n-1]             # GSSG
    
    return Fx

def newton_raphson_solver(x0, X, ab, b, delta_T, CH_bulk, CX_bulk, CY_bulk, Theta, tol=1e-9, max_iter=100):
    """
    Solves the nonlinear system of discretized equations using the Newton-Raphson method.
    Algorithm: x_{k+1} = x_k - J^{-1} * F(x_k)
    Where J is the Jacobian matrix and F is the residual vector.
    """
    # Initialize the residual vector
    Fx = np.zeros_like(x0)

    # Newton-Raphson Iteration Loop
    for _ in range(max_iter):
        
        # 1. Construct the Jacobian Matrix (J)
        ab = compute_jacobian(x0, X, ab, delta_T, CH_bulk, Theta)
        # 2. Compute the Residual Vector (F)
        Fx = compute_Fx(x0, Fx, ab, b, CH_bulk, CX_bulk, CY_bulk)
        # 3. Solve the Linear System (J * dx = -F)
        dx = la.solve_banded((7, 7), ab, -Fx)
        # 4. Update the Solution Estimate
        x0 += dx
        # 5. Check for Convergence
        if np.linalg.norm(dx, np.inf) < tol:
            return x0
        
    return x0

# =============================================================================
# 10. SIMULATION INITIALIZATION
# =============================================================================
flux = []  # List to store dimensionless flux (current)
pot  = []  # List to store dimensionless potential

# Initialize the state vector 'x0' (size 7 * number of nodes)
x0 = np.zeros(7*n)

# Apply initial Bulk Conditions to the entire domain (t=0)
# Slicing [start::step] targets specific species at every node.
x0[1::7] = CB_bulk      # Species 1: Fe2+ (Initially present)
x0[2::7] = CX_bulk      # Species 2: GSH  (Initially present)
x0[3::7] = CY_bulk      # Species 3: GS-  (Initially present)
# Other species (Fe3+, GS*, GSSG*-, GSSG) start at 0.

# 'b' represents the concentration vector at the previous time step (t_k-1)
# Initially, it is identical to the starting conditions.
b = np.copy(x0)
Theta = theta_i     # Set initial dimensionless potential

# =============================================================================
# 11. MAIN TIME-STEPPING LOOP (Cyclic Voltammetry)
# =============================================================================
# Loop over total time steps 'm'
for k in range(m):
    # --- A. Update Applied Potential (Triangle Wave) ---
    if k < m/2:
        Theta += delta_theta
    else:
        Theta -= delta_theta

    # --- B. Solve Nonlinear System for Time t_k ---
    x_solution = newton_raphson_solver(x0, X, ab, b, delta_T, CH_bulk, CX_bulk, CY_bulk, Theta, tol=1e-9, max_iter=100)

    # --- C. Update State Vectors ---
    b = np.copy(x_solution)
    x0 = np.copy(x_solution)

    # --- D. Calculate Current (Flux) ---
    f_BV0 = math.exp(-alpha * Theta)
    flux_A_B = f_BV0 * K1 * (b[1]*math.exp(Theta) - b[0])
    
    # Store results
    flux.append(flux_A_B)
    pot.append(Theta)

# =============================================================================
# 12. DATA POST-PROCESSING AND EXPORT
# =============================================================================
# Convert Dimensionless Flux to Dimensional Current
I = [i*n_e*F*A*DB*cB_bulk*1E6/epsilon for i in flux]
# Convert Dimensionless Potential to Dimensional Voltage
# The -0.235 offset converts the scale to vs. Ag/AgCl
V = [i*R*T/F + Ef - 0.235 for i in pot]

# Stack columns and save to CSV
data_1 = np.column_stack((V, I))
np.savetxt('1_V_2_I_RRD_RSD_combined.txt', data_1, delimiter= ',')

