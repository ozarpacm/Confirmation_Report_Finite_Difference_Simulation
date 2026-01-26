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


