##################################################################################################################################################################################################################
#                                                                     DRONES CONTROL VIA GAN MODEL
#                                                       Participants:
#                                                       Mohamed-Reda Salhi: mohamed-reda.salhi@polytehnique.edu
#                                                       Joseph Combourieu: joseph.combourieu@polytechnique.edu
#                                                       Mohssin Bakraoui : mohssin.bakraoui@polytechnique.edu
#                                                       Andrea Bourelly: andrea.bourelly@polytechnique.edu
#                                                       In collaboration with MBDA
##################################################################################################################################################################################################################





##############################################################
#BLOCK 1: Importations
##############################################################
import sys
import pathlib
import math as m
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Params:
NB_DRONES = 4

TOTAL_TIME = 1

EPSILON = 1e-4
ALPHA_LOSS_G_TERMS = 1.
ALPHA_TARGET = 500.
ALPHA_FORMATION = 70.
ALPHA_OBSTACLE = 1.
ALPHA_COLLISION = 1.
ALPHA_GRAD_PHI = 1.

KABSCH = True


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), skip_weight=0.5):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.skip_weight = skip_weight

    def forward(self, x):
        return self.activation(self.linear(x)) + self.skip_weight * x


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock2 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock3 = ResBlock(hidden_dim, hidden_dim, activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return self.output_layer(x)


# Networks for the GAN-like mean-field game control formulation.
# NOmega approximates the value function (phi network)
class NOmega(nn.Module):
    def __init__(self):
        super(NOmega, self).__init__()
        # Input: 3 (state) + 1 (time) = 4; Output: scalar
        self.net = ResNet(input_dim=4, output_dim=1, activation=nn.Tanh())

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=-1)
        return self.net(input_data)


# NTheta approximates the generator.
class NTheta(nn.Module):
    def __init__(self):
        super(NTheta, self).__init__()
        # Input: 3 (latent) + 1 (time) = 4; Output: 3 (state)
        self.net = ResNet(input_dim=4, output_dim=3, activation=nn.ReLU())

    def forward(self, z, t):
        input_data = torch.cat([z, t], dim=-1)
        return self.net(input_data)

def phi_omega(x, t, N_omega):
    """
    Constructs the value function with boundary condition:
    φ_ω(x, t) = (1 - t) * N_omega(x, t) + t * g(x)
    """
    return (1 - t) * N_omega(x, t) + t * g(x)


def G_theta(z, t, N_theta:NTheta):
    """
    Constructs the generator with boundary condition:
    G_θ(z, t) = (1 - t) * z + t * N_theta(z, t)
    """
    return (1 - t) * z + t * N_theta(z, t)

##############################################################
#BLOCK 2: Costs
##############################################################
'''
SUB-BLOCK: Formation cost
'''
A = 0.1
#the generate wave function is just a way for us to make a list of positions in the form  of a wave for the generate_density function bu can be changed by just List [N,3]
#that  has the positions you wish to have
def generate_wave(n_samples) :
    # k = 2*m.pi/0.5
    # x = torch.linspace(-0.5, 0.5, n_samples, device=device)
    # y = A * torch.sin(k * x)
    # z = torch.ones_like(x, device=device) * 0
    # return torch.stack([x, y, z], dim=1)
    k = 2*m.pi/0.5
    x = torch.linspace(-1/4, 1/4, n_samples, device=device)
    y = torch.ones_like(x, device=device) * (-1/2)
    z = torch.zeros_like(x, device=device)
    return torch.stack([x, y, z], dim=1)

variance = 0.003

def generate_density(x) :
    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    return density_estimated

initial_positions = generate_wave(NB_DRONES)
density_real = generate_density(initial_positions)

def sample_from_density(density_func, n_samples, bounds=(-1, 1), M=None):
    """Méthode du rejet pour échantilloner selon la densité"""
    if M is None:
        # estimer M grossièrement
        pts = (torch.rand(10000, 3, device=device) * (bounds[1]-bounds[0]) + bounds[0])
        M = density_func(pts).max().item() * 1.2 + 1e-9
    samples = []
    while len(samples) < n_samples:
        batch = (torch.rand(n_samples, 3, device=device) * (bounds[1]-bounds[0]) + bounds[0])
        u = torch.rand(n_samples, device=device) * M
        keep = u <= density_func(batch)
        samples.extend(batch[keep].split(1))
    return torch.cat(samples[:n_samples], dim=0)

def distance_L1_torch(p_func, q_func, n_grid, a=-1.0, b=2.0, device=device):
    coords = torch.linspace(a, b, n_grid, device=device)
    dx = (b - a) / n_grid
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
    flat_grid = grid.view(-1, 3)
    p_vals = p_func(flat_grid)
    q_vals = q_func(flat_grid)
    return torch.sum(torch.abs(p_vals - q_vals)) * dx ** 3

def f_formation_old(x, device=device):
    x = x.to(device)
    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    d = distance_L1_torch(density_real, density_estimated, n_grid=50, device=device)
    return d

# def compute_covariance_matrix(centered_samples):
#     """
#     Calcule la matrice de covariance sur les données CENTRÉES.
#     """
#     # Échantillonnage uniforme :
#     # samples = torch.rand(num_samples, 3) * (limits[1] - limits[0]) + limits[0]
    
#     # barycentre selon la valeur de la densité en chaque point :
#     # densities = density_func(samples)
#     # weights = densities / torch.sum(densities)
#     # weights_2d = weights.unsqueeze(1).repeat(1, 3)
#     # m = torch.sum(samples) / len(samples)

#     # soustraction du barycentre :
#     # centered_samples = samples - m
    
#     # Matrice de covariance sur données centrées
#     cov_mat = torch.zeros((3, 3), device=device)
#     for i in range(3):
#         for j in range(3):
#             cov_mat[i,j] = torch.sum(centered_samples[:,i] * centered_samples[:,j])
    
#     return cov_mat

# Pour comparer deux formations :
# def get_sorted_eigenvalues(cov_matrix):
#     """
#     Retourne les valeurs propres d'une matrice de covariance triées par ordre croissant.
#     """
#     eigenvalues = torch.linalg.eigvals(cov_matrix).double().to(device)
#     eigenvalues.sort()
#     return eigenvalues[0]

# def eigen_distance_squared(eigenvalues1, eigenvalues2):
#     """
#     Calcule la distance entre deux listes de valeurs propres triées.
#     """
#     return torch.sum((eigenvalues1 - eigenvalues2)**2)

# def compare_densities(density_func1, density_func2):
#     cov_mat_1 = compute_covariance_matrix_centered(density_func1, limits=(-10, 10), num_samples=500000)
#     cov_mat_2 = compute_covariance_matrix_centered(density_func2, limits=(-10, 10), num_samples=500000)
#     eigenvalues_1 = get_sorted_eigenvalues(cov_mat_1)
#     eigenvalues_2 = get_sorted_eigenvalues(cov_mat_2)

#     return eigen_distance_squared(eigenvalues_1, eigenvalues_2)

# def f_formation_eigenvalues(sample_x, sample_x_pushforwarded):
#     x1 = sample_x.to(device)
#     x2 = sample_x_pushforwarded.to(device)

#     x1_centered = x1 - x1.mean(dim=0, keepdim=True)
#     x2_centered = x2 - x2.mean(dim=0, keepdim=True)

#     cov_mat_1 = compute_covariance_matrix(x1_centered)
#     cov_mat_2 = compute_covariance_matrix(x2_centered)

#     return eigen_distance_squared(get_sorted_eigenvalues(cov_mat_1), get_sorted_eigenvalues(cov_mat_2))

def kabsch(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    H = x_centered.T @ y_centered
    U, S, V = torch.svd(H) # H = U @ S @ V.T
    R = V @ U.T
    if torch.det(R) < 0:
        V = V.clone()
        V[-1, :] *= -1
    R = V @ U.T
    return R.to(device) # On renvoie la matrice de rotation optimale pour passer de x à y

def umeyama(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
    """
    Renvoie la matrice de rotation et le facteur de scaling tels que :
    c * x_c @ R2.T = y_c
    (où x_c et y_x sont les nuages x et y recentrés)
    """
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    n, d = x.size()
    H = (y_centered.T @ x_centered) / n

    U, D, Vt = torch.svd(H)
    D = torch.diag(D).to(device)
    U = U.to(device)
    Vt = Vt.to(device)

    S = torch.eye(d).to(device)
    if U.det() * Vt.det() < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt.T


    sigma_x2 = (x_centered ** 2).sum() / n
    c = (D @ S).trace() / sigma_x2

    return R.to(device), c # On renvoie la matrice de rotation optimale et le scaling pour passer de x à y

# TODO UMEYAMA
# def umeyama(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
#     x_centered = x - x.mean(dim=0)
#     y_centered = y - y.mean(dim=0)
#     n = x.size()[1]
#     H = x_centered.T @ y_centered
#     U, D, Vt = torch.svd(H)
#     # R = Vt @ U.T
#     S = torch.eye(H.size())
#     if U.det() * Vt.det() < 0:
#         S[-1, -1] = -1

#     R = U @ S @ Vt
#     c = (D @ S).trace()
#     if torch.det(R) < 0:
#         Vt = Vt.clone()
#         Vt[-1, :] *= -1
#     R = Vt @ U.T
#     return R.to(device) # On renvoie la matrice de rotation optimale pour passer de x à y

def f_formation(sample_x, sample_x_pushforwarded, initial_positions_pushforwarded):
    sample_x = sample_x.to(device)
    sample_x_pushforwarded = sample_x_pushforwarded.to(device)
    
    if KABSCH:
        R = kabsch(initial_positions_pushforwarded, initial_positions)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = x_centered @ R.T
    else:
        R, c = umeyama(initial_positions_pushforwarded, initial_positions)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = c * x_centered @ R.T

    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    d = distance_L1_torch(density_real, density_estimated, n_grid=50, device=device)
    return d
'''
SUB-BLOCK: Collision Cost
'''
def f_collision(x_batch):
    x_batch = x_batch.to(device)
    diff = x_batch.unsqueeze(1) - x_batch.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    mask = ~torch.eye(dist_sq.size(0), dtype=torch.bool, device=dist_sq.device)
    dist_sq_no_diag = dist_sq.masked_select(mask).view(dist_sq.size(0), -1)
    loss_matrix = 1.0 / (dist_sq_no_diag + EPSILON)
    if loss_matrix.mean() > 0.03 :
      return torch.tensor(0.0, device=device)
    else :
      return loss_matrix.mean()

'''
SUB-BLOCK: Obstacle Costs
'''
#the list obstacles is the list you can modify to put obstacles however you like here below lies just an example of obstacles that can be configurated
mur_a_passer = [
    [x, 0.4, z]
    for x in torch.linspace(-0.5, -0.2, 3) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [x, 0.4, z]
    for x in torch.linspace(0.2, 0.5, 3) for z in torch.linspace(-0.5, 0.5, 6)
]
boite = [
    [x, 1.5, z]
    for x in torch.linspace(-0.5, 0.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [-0.5, y, z]
    for y in torch.linspace(0.5, 1.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [0.5, y, z]
    for y in torch.linspace(0.5, 1.5, 7) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [x, y, -0.5]
    for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
] + [
    [x, y, 0.5]
    for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
]

OBSTACLE_SIZE = 0.1
# boite = [
#     [
#         -1/2 + i / 7,
#         -1 + j / 10,
#         1/2
#     ]
#     for i in range(7) for j in range(15)
# ] + [
#     [
#         -1/2 + i / 7,
#         -1 + j / 10,
#         -1/2
#     ]
#     for i in range(7) for j in range(15)
# ] + [
#     [
#         -1/2,
#         -1 + j / 10,
#         -1/2 + k / 10
#     ]
#     for j in range(15) for k in range(10)
# ] + [
#     [
#         1/2,
#         -1 + j / 10,
#         -1/2 + k / 10
#     ]
#     for j in range(15) for k in range(10)
# ]

obstacles = mur_a_passer + boite

def f_obstacle(x, obstacles):
    espsilon = 1e-4
    x = x.to(device)
    cost = 0
    batch_size = x.size(0)
    for obstacle in obstacles :
        if not isinstance(obstacle, torch.Tensor):
            obstacle_tensor = torch.tensor(obstacle, device=x.device, dtype=x.dtype)
        else:
            obstacle_tensor = obstacle.to(device)
        for i in range(batch_size):
            Q = torch.norm(x[i] - obstacle_tensor)
            if Q < 2. * OBSTACLE_SIZE: # TODO : 
                cost += 1.0 / (max(Q-OBSTACLE_SIZE, 0) + espsilon) 
    return torch.tensor(cost / batch_size)




##############################################################
#BLOCK 2:  optimization part
##############################################################
# this is part very theoritical I advise you to go read the report 

# def sample_from_wave_density(batch_size):
#     sigma = np.sqrt(variance)
#     k = 2 * m.pi / 0.5
#     x = torch.rand(batch_size, device=device) - 0.5
#     y = A * torch.sin(k * x)
#     z = torch.zeros_like(x, device=device)
#     points = torch.stack([x, y, z], dim=1)
#     noise = torch.randn_like(points, device=device) * sigma
#     return points + noise

def generate_sample(batch_size):
    return torch.rand((batch_size, 3), device=device)*2. - 1  # x dans [-1, 1]



def compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg):
    """
    Computes the loss for the phi network using derivative and collision terms.
    """
    sigma = np.sqrt(variance)
    # Sample latent variables and time (uniform in [0, T])
    z = generate_sample(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    # Generate states using the generator (applied sample-wise)
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )
    # Approximate Laplacian: sum of second order derivatives for each spatial dimension
    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv

    H_phi = torch.norm(grad_phi_x, dim=-1, keepdim=True)
    loss_phi_terms = phi_omega(x, torch.zeros_like(t), N_omega) + grad_phi_t \
                     + (sigma**2 / 2) * laplacian + H_phi
    loss_phi_mean = loss_phi_terms.mean()

    # Regularization term penalizing deviation from the HJB residual.
    HJB_residual = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        HJB_residual[i] = torch.norm(
            # Penser à rajouter f_collision
            grad_phi_t[i] + (sigma**2 / 2)*laplacian[i] + H_phi[i]
        )
    loss_HJB = lambda_reg * HJB_residual.mean()

    return loss_phi_mean + loss_HJB + f_collision(x)


x_target = torch.tensor([0,1,0], device="cpu")
def g(x):
    x = x.to(device)
    return torch.norm(x.mean(dim=0) - x_target.to(device))

def compute_loss_G(N_omega, N_theta, batch_size, T, verbose=False):
    """
    Computes the loss for the generator network.
    """
    sigma = np.sqrt(variance)
    # Sample latent variables and time (uniform in [0, T])
    z = generate_sample(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    phi_val.requires_grad_()
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )

    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv
    
    H_phi = torch.norm(ALPHA_GRAD_PHI * grad_phi_x, dim=-1, keepdim=True)
    loss_G_terms = grad_phi_t + (sigma**2 / 2)*laplacian + H_phi

    x_final = G_theta(z, torch.ones_like(t), N_theta)
    formation_loss = 0
    
    nb_checkpoints = 5
    for i in range(1,nb_checkpoints + 1) :
        sample_x_pushforwarded = G_theta(z, torch.ones_like(t)*i/nb_checkpoints, N_theta)
        with torch.no_grad():
            initial_positions_pushforwarded = G_theta(initial_positions.to(device), torch.ones(NB_DRONES, 1, device=device)*i/nb_checkpoints, N_theta)
        formation_loss += f_formation(z, sample_x_pushforwarded, initial_positions_pushforwarded)
        # formation_loss += f_formation_old(sample_x_pushforwarded, device=device)
    # Penser à rajouter f_collision et f_obstacle
    target_loss = g(x_final)
    # print("target_loss: " + str(target_loss))
    # print(formation_loss/5)

    if verbose:
        print("-------------------------------------------------")
        print(MODEL_NAME)
        print(f"{'collision_loss':20s}", f"{ALPHA_COLLISION * f_collision(x).item():.3f}")
        print(f"{'obstacle_loss':20s}", f"{ALPHA_OBSTACLE * f_obstacle(x,obstacles).item():.3f}")
        print(f"{'formation_loss':20s}", f"{ALPHA_FORMATION*formation_loss.item():.3f}")
        print(f"{'target_loss':20s}", f"{ALPHA_TARGET*target_loss.item():.3f}")
        print(f"{'H_phi':20s}", f"{H_phi.mean().item():.3f}")
        print(f"{'loss_G_terms':20s}", f"{ALPHA_LOSS_G_TERMS * loss_G_terms.mean().item():.3f}")
    return target_loss, ALPHA_LOSS_G_TERMS * loss_G_terms.mean() + ALPHA_TARGET*target_loss + ALPHA_FORMATION*formation_loss + ALPHA_OBSTACLE * f_obstacle(x, obstacles) + ALPHA_COLLISION * f_collision(x)





def test_wave_trajectories(n, N_theta, total_time=TOTAL_TIME, num_steps=100):
    """
    For three drones initialized at the vertices of an equilateral triangle,
    generate and plot their trajectories over a total time period (in seconds).

    Args:
        N_theta: Trained generator network (instance of NTheta).
        total_time: Total simulation time (seconds).
        num_steps: Number of time samples along the trajectory.
    """

    # Prepare a list to hold trajectories for each drone
    trajectories = []  # Each entry: NumPy array of shape [num_steps, 3]

    # Generate equally spaced time instants over the total time.
    times = torch.linspace(0, total_time, num_steps, device=device)

    for i in range(n):  # For each drone
        traj = []
        for t_phys in times:
            # Normalize time to [0, 1] for network input
            t_norm = t_phys / total_time
            t_tensor = torch.tensor([[t_norm]], device=device)
            z = initial_positions[i:i+1]  # Shape: [1, 3]
            pos = G_theta(z, t_tensor, N_theta)  # Output: [1, 3]
            traj.append(pos[0])
        traj = torch.stack(traj)  # Shape: [num_steps, 3]
        # Detach before converting to NumPy
        trajectories.append(traj.cpu().detach().numpy())


    # Plot the trajectories
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        traj = trajectories[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        print("Position finale drône " + str(i) + ": " + str(traj[-1]))
    
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]  # résolution de la sphère
    for obs in mur_a_passer:
        cx, cy, cz = float(obs[0]), float(obs[1]), float(obs[2])
        x_s = cx + OBSTACLE_SIZE * np.cos(u) * np.sin(v)
        y_s = cy + OBSTACLE_SIZE * np.sin(u) * np.sin(v)
        z_s = cz + OBSTACLE_SIZE * np.cos(v)
        ax.plot_surface(x_s, y_s, z_s, color='k', alpha=0.6, linewidth=0)
    ax.set_title("Trajectories of " + str(n) + f" Drones Over {total_time: .1f} Seconds")
    ax.plot(x_target[0], x_target[1], x_target[2], 'o', c="yellow")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_box_aspect([1,2,1])
    ax.legend()
    plt.show()


######################################################################

if len(sys.argv) < 11:
    print("usage : python3 main.py <[train / load]> <model_name> <total_time> <epsilon> <alpha_loss_g_terms> <alpha_target> <alpha_formation> <alpha_obstacle> <alpha_collision> <alpha_grad_phi>")
    exit(1)

TOTAL_TIME = float(sys.argv[3])
EPSILON = float(sys.argv[4])
ALPHA_LOSS_G_TERMS = float(sys.argv[5])
ALPHA_TARGET = float(sys.argv[6])
ALPHA_FORMATION = float(sys.argv[7])
ALPHA_OBSTACLE = float(sys.argv[8])
ALPHA_COLLISION = float(sys.argv[9])
ALPHA_GRAD_PHI = float(sys.argv[10])

TRAIN = (sys.argv[1] in ("train", "t"))

PATH = pathlib.Path(sys.argv[0]).resolve().parent
BASE_MODEL_NAME = sys.argv[2]
MODEL_NAME = (
    f"{BASE_MODEL_NAME}_"
    f"T-{TOTAL_TIME}_"
    f"eps-{EPSILON}_"
    f"alphaG-{ALPHA_LOSS_G_TERMS}_"
    f"alphaTarget-{ALPHA_TARGET}_"
    f"alphaForm-{ALPHA_FORMATION}_"
    f"alphaObst-{ALPHA_OBSTACLE}_"
    f"alphaCol-{ALPHA_COLLISION}_"
    f"alphaGradPhi-{ALPHA_GRAD_PHI}"
)
PATH_MODEL_N_OMEGA = PATH / "models" / (MODEL_NAME + "_N_omega")
PATH_MODEL_N_THETA = PATH / "models" / (MODEL_NAME + "_N_theta")


    

def main():
    # Hyperparameters (example values; adjust as needed)
    batch_size = 60
    T = TOTAL_TIME              # Normalized training horizon
    epochs = 2500    # Number of training iterations (increase for convergence)
    lambda_reg = 1.0
    n = NB_DRONES # Nombre de drones

    learning_rate_phi = 4e-4
    learning_rate_gen = 1e-4

    # Instantiate networks and move them to device
    N_omega = NOmega().to(device)
    optimizer_phi = optim.Adam(N_omega.parameters(), lr=learning_rate_phi,
                               betas=(0.5, 0.9), weight_decay=1e-4)
    try:
        checkpoint = torch.load(PATH_MODEL_N_OMEGA, weights_only=True, map_location=torch.device(device=device))
        epoch = checkpoint["epoch"]
        N_omega.load_state_dict(checkpoint["model_state_dict"])
        optimizer_phi.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print(f"Impossible de charger le modèle N_omega : création d'un nouveau modèle")
        epoch=0
    else:
        N_omega.eval()

    N_theta = NTheta().to(device)
    optimizer_theta = optim.Adam(N_theta.parameters(), lr=learning_rate_gen,
                                 betas=(0.5, 0.9), weight_decay=1e-4)
    try:
        checkpoint = torch.load(PATH_MODEL_N_THETA, weights_only=True, map_location=torch.device(device=device))
        N_theta.load_state_dict(checkpoint["model_state_dict"])
        optimizer_theta.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print(f"Impossible de charger le modèle N_theta : création d'un nouveau modèle")
    else:
        N_theta.eval()

    

    # Training loop
    target = 2
    target = 900000
    
    visu = False
    infinite = True
    while TRAIN and (infinite or target > 0.1 or cout > 200):
    # while epoch < epochs:
        optimizer_phi.zero_grad()
        loss_phi_val = compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg)
        loss_phi_val.backward()
        optimizer_phi.step()

        optimizer_theta.zero_grad()
        target, loss_gen_val = compute_loss_G(N_omega, N_theta, batch_size, T, verbose=epoch%10 == 0)
        loss_gen_val.backward()
        cout = loss_gen_val
        optimizer_theta.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss_φ: {loss_phi_val.item():.4f} | Loss_G: {loss_gen_val.item():.4f} | target: {target} | cout {cout}")

        if epoch % 100 == 0:
            print("Sauvegarde des modèles...")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": N_omega.state_dict(),
                        "optimizer_state_dict": optimizer_phi.state_dict()
                    },
                    PATH_MODEL_N_OMEGA
                )
            except:
                print(f"impossible de sauvegarder le fichier {PATH_MODEL_N_OMEGA}")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": N_theta.state_dict(),
                        "optimizer_state_dict": optimizer_theta.state_dict()

                    }, PATH_MODEL_N_THETA)
            except:
                print(f"impossible de sauvegarder le fichier {PATH_MODEL_N_THETA}")

        epoch += 1

        if visu:
            break

    # After training, test by plotting trajectories of n drones over 20 seconds.
    test_wave_trajectories(n, N_theta, total_time=TOTAL_TIME, num_steps=20)


if __name__ == "__main__":
    main()

