from aseams.cvs import CollectiveVariables

def get_x(atoms):
    return float(atoms.positions[0, 0])

R = -0.56756757
P  = 1.677
cv = CollectiveVariables(get_x, get_x, get_x)
cv.set_r_crit("below")
cv.set_in_r_boundary(-0.5)
cv.set_sigma_r_level(-0.48)
cv.set_out_of_r_zone(0.0)
cv.set_p_crit("above")
cv.set_in_p_boundary(P-0.1)