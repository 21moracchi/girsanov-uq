from aseams.cvs import CollectiveVariables

mep_nodes = [
    [-0.558,  1.442], # Minimum 1 (upper left)
    [-0.700,  1.000], # Intermediate
    [-0.822,  0.624], # Transition state 1
    [-0.400,  0.500], # Intermediate
    [-0.050,  0.466], # Minimum 2 (center)
    [ 0.100,  0.350], # Intermediate
    [ 0.212,  0.293], # Transition state 2
    [ 0.400,  0.150], # Intermediate
    [ 0.623,  0.028]  # Minimum 3 (lower right)
]

class PathCollectiveVariable:
    """
    Compute the relative position 's' along a path defined by nodes.
    """
    def __init__(self, path_nodes, lambda_param):
        self.path_nodes = np.array(path_nodes)
        self.lambda_param = lambda_param
        self.N = len(path_nodes)
        # Indices run from 1 to N
        self.indices = np.arange(1, self.N + 1)

    def __call__(self, atoms):
        # Extract only (x, y) from the first atom (r_0, r_1)
        R = atoms.positions[0, :2]
        
        # Squared distances ||R - R_i||^2
        dist_sq = np.sum((self.path_nodes - R)**2, axis=1)
        
        # Exponential terms
        exp_terms = np.exp(-self.lambda_param * dist_sq)
        sum_exp = np.sum(exp_terms)
        
        # Numerical safety: if we are extremely far from the path,
        # return the index of the geometrically nearest node.
        if sum_exp < 1e-50:
            return float(np.argmin(dist_sq) + 1)
            
        # Compute s(R)
        s = np.sum(self.indices * exp_terms) / sum_exp
        return float(s)
mean_dist_sq = np.mean(np.sum(np.diff(mep_nodes, axis=0)**2, axis=1))
lambda_param = 2.3 / mean_dist_sq

# Instantiate the function
s_path_cv = PathCollectiveVariable(path_nodes=mep_nodes, lambda_param=lambda_param)


cv = CollectiveVariables(s_path_cv, s_path_cv, s_path_cv)
cv.set_r_crit("below")
R =1.
P = 9.0
cv.set_in_r_boundary(1.1)
cv.set_sigma_r_level(1.2)
cv.set_out_of_r_zone(2.0)

cv.set_p_crit("above")
cv.set_in_p_boundary(P-0.5)
