from aseams.cvs import CollectiveVariables

def get_x(atoms):
    return float(atoms.positions[0, 0])


cv = CollectiveVariables(get_x, get_x, get_x)
cv.set_r_crit("below")
cv.set_in_r_boundary(-8.0)
cv.set_sigma_r_level(-7.0)
cv.set_out_of_r_zone(-5.0)
cv.set_p_crit("above")
cv.set_in_p_boundary(8.0)
