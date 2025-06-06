import numpy as np
import sympy as sp
import logging
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.linalg import svdvals, eigvals, expm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.special import sph_harm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Current date and time for logging
CURRENT_TIME = datetime(2025, 6, 5, 23, 31)  # 11:31 PM CDT, June 05, 2025

# Physical Constants
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
l_p = np.sqrt(hbar * G / c**3)
m_n = 1.67e-27
RS = 2.0 * G * m_n / c**2

# Configuration
CONFIG = {
    "grid_size": (10, 10, 10, 10, 5, 5),  # (t, x, y, z, v, u)
    "max_iterations": 10,
    "time_delay_steps": 3,
    "ctc_feedback_factor": 0.1,
    "dt": 1e-12,
    "dx": l_p * 1e5,
    "dv": l_p * 1e3,
    "du": l_p * 1e3,
    "omega": 3,
    "a_godel": 1.0,
    "kappa": 1e-8,
    "rtol": 1e-6,
    "atol": 1e-9,
    "field_clamp_max": 1e6,
    "dt_min": 1e-15,
    "dt_max": 1e-9,
    "max_steps_per_dt": 1000,
    "geodesic_steps": 100,
    "ctc_iterations": 20,
    "nugget_m": 1.0,
    "nugget_lambda": 5.0,
    "alpha_time": 3.183e-9,
    "vertex_lambda": 0.33333333326,
    "matrix_size": 100,
}

# Helper Functions
def compute_entanglement_entropy(field, grid_size):
    entropy = np.zeros(grid_size[:4], dtype=np.float64)
    for idx in np.ndindex(grid_size[:4]):
        local_state = field[idx].flatten()
        local_state = np.nan_to_num(local_state, nan=0.0)
        norm = np.linalg.norm(local_state)
        if norm > 1e-15:
            local_state /= norm
        psi_matrix = local_state.reshape(4, 2)
        schmidt_coeffs = svdvals(psi_matrix)
        probs = schmidt_coeffs**2
        probs = probs[probs > 1e-15]
        entropy[idx] = -np.sum(probs * np.log(probs)) if probs.size > 0 else 0
    return np.mean(entropy)

def m_shift(u, v):
    return 2.72

def sample_tetrahedron_points(a, b, c, n_points, m_shift):
    u = np.linspace(-np.pi, np.pi, n_points)
    v = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    u, v = np.meshgrid(u, v)

    faces = []
    # Face 1
    face1_x = a * np.cosh(u) * np.cos(v) * m_shift(u, v)
    face1_y = b * np.cosh(u) * np.sin(v) * m_shift(u, v)
    face1_z = c * np.sinh(u) * m_shift(u, v)
    faces.append((face1_x, face1_y, face1_z))
    # Face 2
    face2_x = -a * np.cosh(u) * np.cos(v) * m_shift(u, v)
    face2_y = -b * np.cosh(u) * np.sin(v) * m_shift(u, v)
    face2_z = c * np.sinh(u) * m_shift(u, v)
    faces.append((face2_x, face2_y, face2_z))
    # Face 3
    face3_x = a * np.cosh(u) * np.cos(v) * m_shift(u, v)
    face3_y = -b * np.cosh(u) * np.sin(v) * m_shift(u, v)
    face3_z = -c * np.sinh(u) * m_shift(u, v)
    faces.append((face3_x, face3_y, face3_z))
    # Face 4
    face4_x = -a * np.cosh(u) * np.cos(v) * m_shift(u, v)
    face4_y = b * np.cosh(u) * np.sin(v) * m_shift(u, v)
    face4_z = -c * np.sinh(u) * m_shift(u, v)
    faces.append((face4_x, face4_y, face4_z))

    points_per_face = 25
    sampled_points = []
    for face_x, face_y, face_z in faces:
        flat_x = face_x.flatten()
        flat_y = face_y.flatten()
        flat_z = face_z.flatten()
        indices = np.linspace(0, len(flat_x) - 1, points_per_face, dtype=int)
        sampled_points.extend([(flat_x[i], flat_y[i], flat_z[i]) for i in indices])
    
    x = np.array([p[0] for p in sampled_points])
    y = np.array([p[1] for p in sampled_points])
    z = np.array([p[2] for p in sampled_points])
    return x, y, z

def hermitian_hamiltonian(x, y, z, k=0.1, J=0.05):
    n = len(x)
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        V_i = k * (x[i]**2 + y[i]**2 + z[i]**2)
        H[i, i] = V_i
        face_size = 25
        pos_in_face = i % face_size
        if pos_in_face < face_size - 1:
            H[i, i + 1] = J
            H[i + 1, i] = J
        face_idx = i // face_size
        if pos_in_face == face_size - 1 and face_idx < 3:
            H[i, i + 1] = J
            H[i + 1, i] = J
    return H

def unitary_matrix(H, t=1.0, hbar=1.0):
    U = expm(-1j * H * t / hbar)
    return U

class TetrahedralLattice:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.deltas = [CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']]
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        dims = [np.linspace(0, self.deltas[i] * size, size, dtype=np.float64)
                for i, size in enumerate(self.grid_size)]
        return np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1)

class NuggetFieldSolver3D:
    def __init__(self, grid_size=30, m=0.1, lambda_ctc=0.5, c=1.0, alpha=0.1, a=1.0, kappa=0.1, g_em=0.3, g_weak=0.65, g_strong=1.0, wormhole_nodes=None):
        self.nx, self.ny, self.nz = grid_size, grid_size, grid_size
        self.grid = np.linspace(-10, 10, grid_size)
        self.x, self.y, self.z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta = np.arccos(self.z / (self.r + 1e-10))
        self.phi = np.arctan2(self.y, self.x)  # Using arctan2 as requested
        self.t_grid = np.linspace(0, 5.0, 100)
        self.dx, self.dt = self.grid[1] - self.grid[0], 0.01
        self.m, self.lambda_ctc, self.c, self.alpha, self.a, self.kappa = m, lambda_ctc, c, alpha, a, kappa
        self.g_em, self.g_weak, self.g_strong = g_em, g_weak, g_strong
        self.phi = np.zeros((self.nx, self.ny, self.nz))
        self.phi_prev = self.phi.copy()
        self.weyl = np.ones((self.nx, self.ny, self.nz))
        self.lambda_harmonic = 2.72
        self.schumann_freq = 7.83
        self.tetrahedral_amplitude = 0.1
        self.wormhole_nodes = wormhole_nodes
        self.ctc_cache = {}
        if self.wormhole_nodes is not None:
            self.precompute_ctc_field()

    def precompute_ctc_field(self):
        for t in self.t_grid:
            ctc_field = np.zeros((self.nx, self.ny, self.nz))
            t_idx = np.argmin(np.abs(self.t_grid - t))
            nodes = self.wormhole_nodes[t_idx, :, :, :, :, :, :]
            x_exp = self.x[..., np.newaxis, np.newaxis]
            y_exp = self.y[..., np.newaxis, np.newaxis]
            z_exp = self.z[..., np.newaxis, np.newaxis]
            x_nodes = nodes[..., 0]
            y_nodes = nodes[..., 1]
            z_nodes = nodes[..., 2]
            distance = np.sqrt((x_exp - x_nodes)**2 + (y_exp - y_nodes)**2 + (z_exp - z_nodes)**2)
            ctc_contrib = np.exp(-distance**2 / (2 * 1.0**2))
            ctc_field = np.mean(ctc_contrib, axis=(3, 4))
            self.ctc_cache[t] = ctc_field
        logger.info("Precomputed CTC field for all time steps.")

    def phi_N_func(self, t, r, theta, phi):
        return np.exp(-r**2) * (1 + self.kappa * np.exp(-t))

    def compute_ricci(self, t):
        phi_N = self.phi_N_func(t, self.r, self.theta, self.phi)
        self.weyl = np.ones_like(self.phi) * (1 + 0.1 * phi_N)
        return self.weyl

    def ctc_function(self, t, x, y, z):
        if self.wormhole_nodes is None:
            u = t * np.pi
            v = t
            phi = 1.618
            C = 2.0
            x_ctc = phi * np.cos(u) * np.sinh(v)
            y_ctc = phi * np.sin(u) * np.sinh(v)
            z_ctc = C * np.cosh(v) * np.cos(u)
            distance = np.sqrt((x - x_ctc)**2 + (y - y_ctc)**2 + (z - z_ctc)**2)
            return np.exp(-distance**2 / (2 * 1.0**2))

        t_closest = min(self.ctc_cache.keys(), key=lambda k: abs(k - t))
        return self.ctc_cache[t_closest]

    def tetrahedral_potential(self, x, y, z):
        vertices = [
            np.array([1, 1, 1]) * self.lambda_harmonic,
            np.array([1, -1, -1]) * self.lambda_harmonic,
            np.array([-1, 1, -1]) * self.lambda_harmonic,
            np.array([-1, -1, 1]) * self.lambda_harmonic
        ]
        min_distance = np.inf
        for vertex in vertices:
            distance = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
            min_distance = np.minimum(min_distance, distance)
        return self.tetrahedral_amplitude * np.exp(-min_distance**2 / (2 * self.lambda_harmonic**2))

    def schumann_potential(self, t):
        return np.sin(2 * np.pi * self.schumann_freq * t)

    def gauge_source(self, t):
        Y_10 = sph_harm(0, 1, self.phi, self.theta)
        source_em = self.g_em * np.sin(t) * np.exp(-self.r) * np.real(Y_10)
        source_weak = self.g_weak * np.cos(t) * np.exp(-self.r) * np.real(Y_10)
        source_strong = self.g_strong * np.ones_like(self.r) * np.real(Y_10)
        return source_em + source_weak + source_strong

    def build_laplacian(self):
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = i * self.ny * self.nz + j * self.nz + k
                    row_ind.append(idx)
                    col_ind.append(idx)
                    data.append(-6 / self.dx**2)
                    for di in [-1, 1]:
                        ni = i + di
                        if 0 <= ni < self.nx:
                            row_ind.append(idx)
                            col_ind.append(ni * self.ny * self.nz + j * self.nz + k)
                            data.append(1 / self.dx**2)
                    for dj in [-1, 1]:
                        nj = j + dj
                        if 0 <= nj < self.ny:
                            row_ind.append(idx)
                            col_ind.append(i * self.ny * self.nz + nj * self.nz + k)
                            data.append(1 / self.dx**2)
                    for dk in [-1, 1]:
                        nk = k + dk
                        if 0 <= nk < self.nz:
                            row_ind.append(idx)
                            col_ind.append(i * self.ny * self.nz + j * self.nz + nk)
                            data.append(1 / self.dx**2)

        laplacian = csr_matrix((data, (row_ind, col_ind)), shape=(self.nx * self.ny * self.nz, self.nx * self.ny * self.nz))
        return laplacian

    def effective_mass(self):
        return self.m**2 * (1 + self.alpha * np.mean(self.weyl))

    def rhs(self, t, phi_flat):
        phi = phi_flat.reshape((self.nx, self.ny, self.nz))
        self.phi_prev = self.phi.copy()
        self.phi = phi
        phi_t = (phi - self.phi_prev) / self.dt
        laplacian = self.build_laplacian().dot(phi_flat).reshape(self.nx, self.ny, self.nz)
        ctc_term = self.lambda_ctc * self.ctc_function(t, self.x, self.y, self.z) * phi
        source = self.gauge_source(t)
        self.compute_ricci(t)
        tetrahedral_term = self.tetrahedral_potential(self.x, self.y, self.z) * phi
        schumann_term = self.schumann_potential(t) * phi
        return (phi_t / self.dt + self.c**-2 * phi_t + laplacian - self.effective_mass() * phi + ctc_term - source + tetrahedral_term + schumann_term).flatten()

    def solve(self, t_end=5.0, nt=500):
        t_values = np.linspace(0, t_end, nt)
        initial_state = self.phi.flatten()
        sol = solve_ivp(self.rhs, [0, t_end], initial_state, t_eval=t_values, method='RK45')
        self.phi = sol.y[:, -1].reshape((self.nx, self.ny, self.nz))
        return self.phi

class Unified6DSimulation:
    def __init__(self):
        self.grid_size = CONFIG["grid_size"]
        self.total_points = np.prod(self.grid_size)
        self.dt = CONFIG["dt"]
        self.deltas = [CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']]
        self.time = 0.0

        self.lattice = TetrahedralLattice(self.grid_size)
        self.wormhole_nodes, self.wormhole_signal = self._generate_wormhole_nodes()

        self.quantum_state = np.ones(self.grid_size, dtype=np.complex128) / np.sqrt(self.total_points)
        self.phi_N = np.zeros(self.grid_size, dtype=np.float64)
        self.stress_energy = self._initialize_stress_energy()

        self.ctc_state = np.zeros(16, dtype=np.complex128)
        self.ctc_state[0] = 1.0

        self.nugget_solver = NuggetFieldSolver3D(grid_size=30, m=CONFIG["nugget_m"], lambda_ctc=CONFIG["nugget_lambda"], wormhole_nodes=self.wormhole_nodes)
        self.nugget_field = np.zeros((30, 30, 30))

        self.setup_symbolic_calculations()
        self.metric, self.inverse_metric = self.compute_quantum_metric()
        self.connection = self._compute_affine_connection()
        self.riemann_tensor = self._compute_riemann_tensor()
        self.ricci_tensor, self.ricci_scalar = self._compute_curvature()
        self.einstein_tensor = self._compute_einstein_tensor()

        self.tetrahedral_nodes, self.napoleon_centroids = self._generate_enhanced_tetrahedral_nodes()
        self.ctc_unitary = self._compute_ctc_unitary_matrix()

        self.geodesic_path = None

        self.history = []
        self.phi_N_history = []
        self.nugget_field_history = []
        self.result_history = []
        self.ctc_state_history = []
        self.entanglement_history = []
        self.time_displacement_history = []

    def setup_symbolic_calculations(self):
        self.t, self.x, self.y, self.z, self.v, self.u = sp.symbols('t x y z v u')
        self.a, self.c_sym, self.m, self.kappa_sym = sp.symbols('a c m kappa', positive=True)
        self.phi_N_sym = sp.Function('phi_N')(self.t, self.x, self.y, self.z, self.v, self.u)
        r = sp.sqrt(self.x**2 + self.y**2 + self.z**2 + 1e-10)
        theta = sp.atan2(self.y, self.x)
        phi = sp.atan2(self.z, sp.sqrt(self.x**2 + self.y**2))

        scaling_factor = 1.151287

        g_tt = scaling_factor * (-self.c_sym**2 * (1 + self.kappa_sym * self.phi_N_sym))
        g_rr = scaling_factor * (self.a**2 * sp.exp(2 * r / self.a) * (1 + self.kappa_sym * self.phi_N_sym))
        g_theta_theta = scaling_factor * (self.a**2 * (sp.exp(2 * r / self.a) - 1) * (1 + self.kappa_sym * self.phi_N_sym))
        g_tphi = scaling_factor * (self.a * self.c_sym * sp.exp(r / self.a))
        g_phi_phi = scaling_factor * (1 + self.kappa_sym * self.phi_N_sym)

        self.g = sp.zeros(6, 6)
        self.g[0, 0] = g_tt
        self.g[1, 1] = g_rr
        self.g[2, 2] = g_theta_theta
        self.g[3, 3] = g_phi_phi
        self.g[0, 3] = self.g[3, 0] = g_tphi
        self.g[4, 4] = scaling_factor * (l_p**2)
        self.g[5, 5] = scaling_factor * (l_p**2)

        self.g_inv = self.g.inv()
        self.metric_scale_factors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def _generate_wormhole_nodes(self):
        coords = np.zeros((*self.grid_size, 6), dtype=np.float64)
        signal = np.zeros(self.grid_size, dtype=np.int64)
        dims = [np.linspace(0, self.deltas[i] * size, size, dtype=np.float64)
                for i, size in enumerate(self.grid_size)]
        T, X, Y, Z, V, U = np.meshgrid(*dims, indexing='ij')
        phi = 1.618
        C = 2.0
        alpha_time = CONFIG["alpha_time"]
        coords[..., 0] = phi * np.cos(U) * np.sinh(V)
        coords[..., 1] = phi * np.sin(U) * np.sinh(V)
        coords[..., 2] = C * np.cosh(V) * np.cos(U)
        coords[..., 5] = alpha_time * 2 * np.pi * C * np.cosh(V) * np.sin(U)
        R, r = 1.5 * self.deltas[1], 0.5 * self.deltas[1]
        coords[..., 3] = r * np.cos(CONFIG["omega"] * V)
        coords[..., 4] = r * np.sin(CONFIG["omega"] * U)
        return np.nan_to_num(coords, nan=0.0), signal

    def _barycentric_interpolation(self, p1, p2, p3, weights):
        w1, w2, w3 = weights
        x = w1 * p1[0] + w2 * p2[0] + w3 * p3[0]
        y = w1 * p1[1] + w2 * p2[1] + w3 * p3[1]
        z = w1 * p1[2] + w2 * p2[2] + w3 * p3[2]
        return np.array([x, y, z])

    def _napoleon_centroid(self, p1, p2, p3):
        mid12 = (p1 + p2) / 2
        mid23 = (p2 + p3) / 2
        mid31 = (p3 + p1) / 2
        v12 = p2 - p1
        v23 = p3 - p2
        v31 = p1 - p3
        centroid = (p1 + p2 + p3) / 3
        return centroid

    def _generate_enhanced_tetrahedral_nodes(self):
        lambda_vertex = CONFIG["vertex_lambda"]
        a, b, c = 1, 2, 3
        n_points = 100
        x, y, z = sample_tetrahedron_points(a, b, c, n_points, m_shift)
        points = np.stack([x, y, z], axis=-1)

        face_indices = [
            list(range(0, 25)),
            list(range(25, 50)),
            list(range(50, 75)),
            list(range(75, 100))
        ]

        selected_points = []
        face_centroids = []
        napoleon_centroids = []
        for face_idx in face_indices:
            face_points = points[face_idx]
            v1 = face_points[0]
            v2 = face_points[len(face_points)//2]
            v3 = face_points[-1]
            centroid = (v1 + v2 + v3) / 3
            face_centroids.append(centroid)
            nap_centroid = self._napoleon_centroid(v1, v2, v3)
            napoleon_centroids.append(nap_centroid)
            points_to_refine = [v1, v2, v3, centroid]
            refined_points = []
            for p in points_to_refine:
                A = np.array([
                    [v1[0], v2[0], v3[0]],
                    [v1[1], v2[1], v3[1]],
                    [v1[2], v2[2], v3[2]]
                ])
                b = p
                A_aug = np.vstack([A.T, [1, 1, 1]])
                b_aug = np.append(b, 1)
                weights = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]
                weights = np.clip(weights, 0, 1)
                weights /= np.sum(weights)
                refined_p = self._barycentric_interpolation(v1, v2, v3, weights)
                refined_points.append(refined_p)
            selected_points.extend(refined_points)

        selected_points = np.array(selected_points)
        napoleon_centroids = np.array(napoleon_centroids)

        selected_points *= lambda_vertex
        napoleon_centroids *= lambda_vertex

        return selected_points, napoleon_centroids

    def _compute_ctc_unitary_matrix(self):
        a, b, c = 1, 2, 3
        n_points = 100
        x, y, z = sample_tetrahedron_points(a, b, c, n_points, m_shift)
        H = hermitian_hamiltonian(x, y, z, k=0.1, J=0.05)
        U_full = unitary_matrix(H, t=1.0)

        selected_indices = []
        face_size = 25
        for face_start in range(0, 100, face_size):
            indices = [face_start, face_start + face_size//2, face_start + face_size - 1, face_start + face_size//4]
            selected_indices.extend(indices)
        
        U_ctc = U_full[np.ix_(selected_indices, selected_indices)]
        
        identity_check = np.allclose(np.conjugate(U_ctc.T) @ U_ctc, np.eye(16), atol=1e-10)
        logger.info(f"Is U_ctc unitary? {identity_check}")
        
        return U_ctc

    def compute_quantum_metric(self):
        metric = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        coords = self.lattice.coordinates
        for idx in np.ndindex(self.grid_size):
            subs_dict = {
                self.t: coords[idx][0], self.x: coords[idx][1], self.y: coords[idx][2],
                self.z: coords[idx][3], self.v: coords[idx][4], self.u: coords[idx][5],
                self.a: CONFIG["a_godel"], self.c_sym: c, self.m: m_n,
                self.kappa_sym: CONFIG["kappa"], self.phi_N_sym: self.phi_N[idx]
            }
            g = np.array(sp.N(self.g.subs(subs_dict)), dtype=np.float64)
            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                logger.warning(f"NaN/Inf detected in metric at idx {idx}: {g}")
                g = np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
            metric[idx] = 0.5 * (g + g.T)
        inverse_metric = np.linalg.pinv(metric, rcond=1e-10)
        scale_matrix = np.diag(1.0 / self.metric_scale_factors)
        for idx in np.ndindex(self.grid_size):
            inverse_metric[idx] = scale_matrix @ inverse_metric[idx] @ scale_matrix
        return np.nan_to_num(metric, nan=0.0), np.nan_to_num(inverse_metric, nan=0.0)

    def _compute_affine_connection(self):
        connection = np.zeros((*self.grid_size, 6, 6, 6), dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if all(0 < i < s - 1 for i, s in zip(idx, self.grid_size)):
                for rho in range(6):
                    for mu in range(6):
                        for nu in range(6):
                            dg_mu_nu = np.gradient(self.metric[..., mu, nu], self.deltas[rho], axis=rho)[idx]
                            dg_rho_mu = np.gradient(self.metric[..., rho, mu], self.deltas[nu], axis=nu)[idx]
                            dg_rho_nu = np.gradient(self.metric[..., rho, nu], self.deltas[mu], axis=mu)[idx]
                            term = dg_mu_nu + dg_rho_mu - dg_rho_nu
                            connection[idx][rho, mu, nu] = 0.5 * np.einsum('rs,s->r', self.inverse_metric[idx], term)
        return np.nan_to_num(connection, nan=0.0)

    def _compute_riemann_tensor(self):
        riemann = np.zeros((*self.grid_size, 6, 6, 6, 6), dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if all(0 < i < s - 1 for i, s in zip(idx, self.grid_size)):
                for rho in range(6):
                    for sigma in range(6):
                        for mu in range(6):
                            for nu in range(6):
                                dGamma_nu = np.gradient(self.connection[idx][rho, nu, sigma], self.deltas[mu])
                                dGamma_mu = np.gradient(self.connection[idx][rho, mu, sigma], self.deltas[nu])
                                term1 = np.einsum('km,mn->kn', self.connection[idx][rho, :, mu],
                                                  self.connection[idx][:, nu, sigma])
                                term2 = np.einsum('kn,mn->km', self.connection[idx][rho, :, nu],
                                                  self.connection[idx][:, mu, sigma])
                                riemann[idx][rho, sigma, mu, nu] = dGamma_nu - dGamma_mu + term1 - term2
        max_val = np.max(np.abs(riemann))
        if max_val > 1e5:
            riemann /= max_val
            logger.info(f"Riemann tensor scaled by {max_val} to prevent overflow")
        return np.nan_to_num(riemann, nan=0.0)

    def _compute_curvature(self):
        ricci_tensor = np.einsum('...rsmn,mn->...rs', self.riemann_tensor, self.inverse_metric)
        ricci_scalar = np.einsum('...mn,mn->...', self.inverse_metric, ricci_tensor)
        return np.nan_to_num(ricci_tensor, nan=0.0), np.nan_to_num(ricci_scalar, nan=0.0)

    def _compute_einstein_tensor(self):
        ricci_tensor, ricci_scalar = self._compute_curvature()
        einstein_tensor = ricci_tensor - 0.5 * self.metric * ricci_scalar[..., np.newaxis, np.newaxis]
        return np.nan_to_num(einstein_tensor, nan=0.0)

    def _initialize_stress_energy(self):
        T = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        T_base = np.zeros((6, 6), dtype=np.float64)
        T_base[0, 0] = 3.978873e-12
        spatial_T = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float64)
        T_base[1:4, 1:4] = spatial_T
        for idx in np.ndindex(self.grid_size):
            T[idx] = T_base
        return np.nan_to_num(T, nan=0.0)

    def update_stress_energy_with_nugget(self):
        T_nugget = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if all(0 < i < s - 1 for i, s in zip(idx, self.grid_size)):
                phi = self.nugget_field[idx[1], idx[2], idx[3]] if idx[1] < 30 and idx[2] < 30 and idx[3] < 30 else 0
                partial_phi = np.zeros(6)
                for mu in range(6):
                    idx_plus = list(idx)
                    idx_minus = list(idx)
                    idx_plus[mu] += 1
                    idx_minus[mu] -= 1
                    if 0 <= idx_plus[mu] < self.grid_size[mu] and 0 <= idx_minus[mu] < self.grid_size[mu]:
                        phi_plus = self.nugget_field[idx_plus[1], idx_plus[2], idx_plus[3]] if idx_plus[1] < 30 and idx_plus[2] < 30 and idx_plus[3] < 30 else 0
                        phi_minus = self.nugget_field[idx_minus[1], idx_minus[2], idx_minus[3]] if idx_minus[1] < 30 and idx_minus[2] < 30 and idx_minus[3] < 30 else 0
                        partial_phi[mu] = (phi_plus - phi_minus) / (2 * self.deltas[mu])
                partial_term = 0
                for rho in range(6):
                    for sigma in range(6):
                        partial_term += self.inverse_metric[idx][rho, sigma] * partial_phi[rho] * partial_phi[sigma]
                for mu in range(6):
                    for nu in range(6):
                        T_nugget[idx][mu, nu] = partial_phi[mu] * partial_phi[nu] - 0.5 * self.metric[idx][mu, nu] * (partial_term + CONFIG["nugget_m"]**2 * phi**2)
        self.stress_energy += T_nugget
        self.stress_energy = np.nan_to_num(self.stress_energy, nan=0.0)

    def solve_geodesic(self, initial_position, initial_velocity):
        def geodesic_equation(tau, y):
            x = y[:6]
            dx_dtau = y[6:]
            d2x_dtau2 = np.zeros(6)
            idx = tuple(np.clip(int(x[i] / self.deltas[i]), 0, s-1) for i, s in zip(range(6), self.grid_size))
            Gamma = self.connection[idx]
            for lam in range(6):
                for mu in range(6):
                    for nu in range(6):
                        d2x_dtau2[lam] -= Gamma[lam, mu, nu] * dx_dtau[mu] * dx_dtau[nu]
            return np.concatenate([dx_dtau, d2x_dtau2])

        y0 = np.concatenate([initial_position, initial_velocity])
        tau_span = (0, 1)
        sol = solve_ivp(geodesic_equation, tau_span, y0, method='RK45',
                        t_eval=np.linspace(0, 1, CONFIG["geodesic_steps"]),
                        rtol=CONFIG["rtol"], atol=CONFIG["atol"])
        if not sol.success:
            raise ValueError("Geodesic equation solver failed")
        self.geodesic_path = sol.y[:6].T
        return self.geodesic_path

    def compute_time_displacement(self, u_entry, u_exit, v=0):
        C = 2.0
        alpha_time = CONFIG["alpha_time"]
        t_entry = alpha_time * 2 * np.pi * C * np.cosh(v) * np.sin(u_entry)
        t_exit = alpha_time * 2 * np.pi * C * np.cosh(v) * np.sin(u_exit)
        return t_exit - t_entry

    def adjust_time_displacement(self, target_dt, u_entry=np.pi/2, v=0):
        C = 2.0
        alpha_time = CONFIG["alpha_time"]
        def objective(delta_u):
            u_exit = u_entry + delta_u
            dt = self.compute_time_displacement(u_entry, u_exit, v)
            return (dt - target_dt)**2

        result = minimize(objective, x0=0.01, method='Nelder-Mead', tol=1e-15)
        delta_u = result.x[0]
        u_exit = u_entry + delta_u
        actual_dt = self.compute_time_displacement(u_entry, u_exit, v)
        logger.info(f"Target time displacement: {target_dt:.6e}, Achieved: {actual_dt:.6e}, u_exit: {u_exit:.6f}")
        return u_exit, actual_dt

    def transmit_and_compute(self, input_data, direction="future", target_dt=None):
        if target_dt is None:
            target_dt = self.dt if direction == "future" else -self.dt

        entry_time = self.time
        entry_u = np.pi / 2
        entry_coords = (entry_time, 0, 0, 0, 0, entry_u)
        entry_idx = tuple(int(coord / delta) % size for coord, delta, size in zip(entry_coords, self.deltas, self.grid_size))

        u_exit, actual_dt = self.adjust_time_displacement(target_dt, u_entry=entry_u, v=0)
        adjusted_time = self.time + actual_dt
        exit_coords = (adjusted_time, 0, 0, 0, 0, u_exit)
        exit_idx = tuple(int(coord / delta) % size for coord, delta, size in zip(exit_coords, self.deltas, self.grid_size))

        entry_node_coords = self.wormhole_nodes[entry_idx]
        exit_node_coords = self.wormhole_nodes[exit_idx]
        entry_node_time = entry_node_coords[5]
        exit_node_time = exit_node_coords[5]

        logger.info(f"Entry node time (from nodes): {entry_node_time:.6e}, Simulation time: {entry_time:.6e}")
        logger.info(f"Exit node time (from nodes): {exit_node_time:.6e}, Target time: {adjusted_time:.6e}")

        time_label = "future" if actual_dt > 0 else "past"
        if (actual_dt > 0 and direction == "past") or (actual_dt < 0 and direction == "future"):
            logger.warning(f"Time displacement direction mismatch: expected {direction}, got {time_label}")

        initial_position = np.array(entry_coords)
        initial_velocity = (np.array(exit_coords) - np.array(entry_coords)) / 1.0
        geodesic_path = self.solve_geodesic(initial_position, initial_velocity)
        logger.info(f"Geodesic path computed from {entry_coords} to {exit_coords}")

        plt.figure(figsize=(8, 6))
        plt.plot(geodesic_path[:, 5], geodesic_path[:, 0], label=f'Geodesic Path (u vs t, to {time_label})')
        plt.xlabel('u')
        plt.ylabel('t')
        plt.title(f'Geodesic Path Through Wormhole (To the {time_label.capitalize()})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'geodesic_path_to_{time_label}.png')
        plt.close()
        logger.info(f"Geodesic path plot saved as 'geodesic_path_to_{time_label}.png'")

        self.wormhole_signal[entry_idx] = input_data
        logger.info(f"Encoded input {input_data} at entry node {entry_idx} (coords: {entry_coords})")

        self.wormhole_signal[exit_idx] = self.wormhole_signal[entry_idx]
        logger.info(f"Transmitted input to exit node {exit_idx} (coords: {exit_coords}) at {time_label} time {exit_node_time:.6e}")

        matrix_size = CONFIG["matrix_size"]
        A = np.random.randn(matrix_size, matrix_size) + 1j * np.random.randn(matrix_size, matrix_size)
        A = (A + A.conj().T) / 2
        eigenvalues = eigvals(A)
        result = np.sum(np.abs(eigenvalues))
        logger.info(f"Computed result at {time_label} node (time {exit_node_time:.6e}): Eigenvalue sum = {result:.6e}")

        self.wormhole_signal[exit_idx] = int(result % 1000000)
        self.wormhole_signal[entry_idx] = self.wormhole_signal[exit_idx]
        logger.info(f"Retrieved result {result:.6e} at entry node {entry_idx} at present time {entry_node_time:.6e}")

        self._propagate_causal_effects(entry_idx, exit_idx, input_data, result, actual_dt)

        return result

    def _propagate_causal_effects(self, entry_idx, exit_idx, input_data, result, time_displacement):
        influence_factor = (result % 1000) / 1000.0
        self.quantum_state[entry_idx] *= np.exp(1j * influence_factor * np.pi)
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
        logger.info(f"Propagated causal effects: adjusted quantum state at entry_idx {entry_idx} with influence factor {influence_factor:.6f}")

    def simulate_ctc_quantum_circuit(self):
        num_qubits = 4
        dim = 2**num_qubits
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        H_0 = np.kron(H, np.kron(I, np.kron(I, I)))
        H_1 = np.kron(I, np.kron(H, np.kron(I, I)))
        CNOT_01 = np.kron(CNOT, np.kron(I, I))
        CNOT_12 = np.kron(I, np.kron(CNOT, I))
        CNOT_23 = np.kron(I, np.kron(I, CNOT))
        U_ctc = self.ctc_unitary
        U_cycle = H_0 @ CNOT_01 @ H_1 @ CNOT_12 @ CNOT_23 @ U_ctc
        def consistency_loss(params):
            real = params[:dim]
            imag = params[dim:]
            state = real + 1j * imag
            norm = np.linalg.norm(state)
            if norm > 0:
                state /= norm
            state_out = U_cycle @ state
            diff = state_out - state
            return np.sum(np.abs(diff)**2)

        initial_guess = np.concatenate([self.ctc_state.real, self.ctc_state.imag])
        result = minimize(consistency_loss, initial_guess, method='L-BFGS-B', tol=1e-10)
        optimized_params = result.x
        state = optimized_params[:dim] + 1j * optimized_params[dim:]
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        probs = np.abs(state)**2
        max_prob_idx = np.argmax(probs)
        if probs[max_prob_idx] > 0.5:
            lambda_vertex = CONFIG["vertex_lambda"]
            control_phase = np.exp(1j * lambda_vertex * np.pi * max_prob_idx)
            control_gate = np.diag([control_phase if i == max_prob_idx else 1.0 for i in range(dim)])
            state = control_gate @ state
            norm = np.linalg.norm(state)
            if norm > 0:
                state /= norm
            probs = np.abs(state)**2
        prob_0 = sum(probs[:dim//2])
        prob_1 = sum(probs[dim//2:])
        decision = 0 if prob_0 > prob_1 else 1
        self.ctc_state = state
        logger.info(f"CTC quantum circuit state probabilities: {probs}")
        logger.info(f"CTC quantum circuit decision (cyclical self-consistent model): {decision}")
        return decision

    def evolve_quantum_state(self, dt):
        def quantum_deriv(t, q_flat):
            q = q_flat.reshape(self.grid_size)
            kinetic = np.sum([np.gradient(np.gradient(q, self.deltas[mu], axis=mu), self.deltas[mu], axis=mu)
                              for mu in range(6)], axis=0)
            potential = self.phi_N * q
            return (-hbar**2 / (2 * m_n) * kinetic + potential).flatten()

        q_flat = self.quantum_state.flatten()
        sol = solve_ivp(quantum_deriv, [0, dt], q_flat, method='RK45',
                        rtol=CONFIG["rtol"], atol=CONFIG["atol"])
        if not sol.success:
            raise ValueError("solve_ivp failed in evolve_quantum_state")
        self.quantum_state = sol.y[:, -1].reshape(self.grid_size)
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
        self.quantum_state = np.clip(self.quantum_state, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        return len(sol.t) - 1

    def adjust_time_step(self, steps_taken):
        target_steps = 10
        if steps_taken > CONFIG["max_steps_per_dt"]:
            self.dt *= 0.5
            logger.info(f"Reducing dt to {self.dt} due to excessive steps ({steps_taken})")
        elif steps_taken > target_steps * 1.5:
            self.dt *= 0.9
            logger.debug(f"Reducing dt to {self.dt} (steps: {steps_taken})")
        elif steps_taken < target_steps * 0.5 and steps_taken > 0:
            self.dt *= 1.1
            logger.debug(f"Increasing dt to {self.dt} (steps: {steps_taken})")
        self.dt = max(CONFIG["dt_min"], min(self.dt, CONFIG["dt_max"]))

    def visualize_tetrahedral_nodes(self):
        nodes = self.tetrahedral_nodes
        centroids = self.napoleon_centroids
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        ax.scatter(x, y, z, c='b', marker='o', label='Nodes')
        cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        ax.scatter(cx, cy, cz, c='r', marker='^', label='Napoleon Centroids')
        face_indices = [
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
        ]
        for face_idx, face in enumerate(face_indices):
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    idx1, idx2 = face[i] + face_idx*4, face[j] + face_idx*4
                    ax.plot([nodes[idx1][0], nodes[idx2][0]], [nodes[idx1][1], nodes[idx2][1]], [nodes[idx1][2], nodes[idx2][2]], 'b-')
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                ax.plot([centroids[i][0], centroids[j][0]], [centroids[i][1], centroids[j][1]], [centroids[i][2], centroids[j][2]], 'r--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Enhanced Tetrahedral Lattice of 16 CTC Wormhole Nodes')
        plt.legend()
        plt.savefig('tetrahedral_nodes_enhanced.png')
        plt.close()
        logger.info("Enhanced tetrahedral nodes plot saved as 'tetrahedral_nodes_enhanced.png'")

    def run_simulation(self):
        logger.info(f"Starting simulation with scaling factor: 1.151287")
        self.visualize_tetrahedral_nodes()

        for iteration in range(CONFIG["max_iterations"]):
            t_start = self.time
            t_end = t_start + self.dt
            max_attempts = 3
            current_dt = self.dt
            total_steps = 0

            for attempt in range(max_attempts):
                try:
                    self.time = t_start

                    self.nugget_field = self.nugget_solver.solve(t_end=self.time + current_dt, nt=2)
                    logger.info(f"Nugget Field mean value: {np.mean(self.nugget_field):.6e}")

                    self.update_stress_energy_with_nugget()

                    input_data = 1011
                    target_dt_past = -self.dt * 2
                    result_past = self.transmit_and_compute(input_data, direction="past", target_dt=target_dt_past)
                    logger.info(f"Result from past computation: {result_past:.6e}")

                    result_future = self.transmit_and_compute(input_data, direction="future", target_dt=self.dt * 2)
                    logger.info(f"Result from future computation (speed-up): {result_future:.6e}")

                    ctc_decision = self.simulate_ctc_quantum_circuit()
                    logger.info(f"CTC quantum circuit decision (cyclical self-consistent model): {ctc_decision}")

                    steps = self.evolve_quantum_state(current_dt)
                    total_steps += steps

                    self.metric, self.inverse_metric = self.compute_quantum_metric()
                    self.connection = self._compute_affine_connection()
                    self.riemann_tensor = self._compute_riemann_tensor()
                    self.ricci_tensor, self.ricci_scalar = self._compute_curvature()
                    self.einstein_tensor = self._compute_einstein_tensor()

                    entanglement_entropy = compute_entanglement_entropy(self.quantum_state, self.grid_size)

                    sample_idx = (0, 1, 1, 1, 0, 0)
                    G_00 = self.einstein_tensor[sample_idx][0, 0]
                    T_00 = self.stress_energy[sample_idx][0, 0]
                    discrepancy = abs(G_00 - 8 * np.pi * T_00)
                    logger.info(f"Iteration {iteration}: G_00 = {G_00:.6e}, 8pi T_00 = {8 * np.pi * T_00:.6e}, "
                                f"Discrepancy = {discrepancy:.6e}, Past Result = {result_past:.6e}, "
                                f"Future Result = {result_future:.6e}, CTC Decision = {ctc_decision}, "
                                f"Entanglement Entropy = {entanglement_entropy:.6e}")

                    self.history.append(self.time)
                    self.phi_N_history.append(self.phi_N[0, 0, 0, 0, 0, 0])
                    self.nugget_field_history.append(np.mean(self.nugget_field))
                    self.result_history.append((result_past, result_future))
                    self.ctc_state_history.append(self.ctc_state.copy())
                    self.entanglement_history.append(entanglement_entropy)

                    self.time = t_end
                    break
                except Exception as e:
                    logger.warning(f"Simulation failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        raise
                    current_dt *= 0.5
                    logger.info(f"Retrying with reduced dt={current_dt}")
                    self.dt = current_dt

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.entanglement_history)), self.entanglement_history, label='Entanglement Entropy')
        plt.xlabel('Iteration')
        plt.ylabel('Entanglement Entropy')
        plt.title('Entanglement Entropy Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig('entanglement_entropy.png')
        plt.close()
        logger.info("Entanglement entropy plot saved as 'entanglement_entropy.png'")

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.nugget_field_history)), self.nugget_field_history, label='Nugget Field Mean')
        plt.xlabel('Iteration')
        plt.ylabel('Nugget Field Mean Value')
        plt.title('Nugget Field Evolution Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig('nugget_field_evolution.png')
        plt.close()
        logger.info("Nugget field evolution plot saved as 'nugget_field_evolution.png'")

if __name__ == "__main__":
    sim = Unified6DSimulation()
    sim.run_simulation()
    logger.info("Simulation completed successfully.")