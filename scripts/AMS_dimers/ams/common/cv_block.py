from girsanov_uq.potentials.dimer_potential import SolvatedDimer

def get_negative_d_AB(atoms):
    # Use MIC so the CV is consistent with periodic boundary conditions.
    return -atoms.get_distance(0, 1, mic=True)


cv = CollectiveVariables(get_negative_d_AB, get_negative_d_AB, get_negative_d_AB)
cv.set_r_crit("below")
R = 1.622462048309373
P = 1.122462048309373
cv.set_in_r_boundary(-R+0.03)
cv.set_sigma_r_level(-R+0.04)
cv.set_out_of_r_zone(-(P+R)/2)

cv.set_p_crit("above")
cv.set_in_p_boundary(-P-0.1)
