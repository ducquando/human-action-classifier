import torch
import torch.nn.functional as F

# -------------------------------------------------------------
# Gaussian Scale Space
# -------------------------------------------------------------

def gaussian_1d(kernel_size, sigma):
    """
    Generate a 1D Gaussian kernel.

    Args:
        kernel_size (int)
        sigma (float)
    """
    x = torch.arange(kernel_size) - kernel_size // 2
    g = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
    return g / g.sum()


def gaussian_blur_3d(video, sigma, tau):
    """
    Apply separable Gaussian smoothing in space and time.

    Args:
        video (Tensor): shape (T,H,W)
        sigma (float): spatial std
        tau (float): temporal std
    """
    # spatial blur
    g_xy = gaussian_1d(7, sigma)
    g_xy = g_xy[None, None, :, None] * g_xy[None, None, None, :]
    g_xy = g_xy.to(video.device)
    video = F.conv2d(video.unsqueeze(1), g_xy, padding=3).squeeze(1)

    # temporal blur
    g_t = gaussian_1d(7, tau)[None, None, :, None, None]
    g_t = g_t.to(video.device)
    video = F.conv3d(video.unsqueeze(0), g_t, padding=(3,0,0)).squeeze(0)

    return video


def gradients_3d(L):
    """
    Compute spatial and temporal gradients.
    Args: L (Tensor): scale-space video
    Returns: Lx, Ly, Lt (Tensor)
    """
    Lx = L[:, :, 2:] - L[:, :, :-2]
    Ly = L[:, 2:, :] - L[:, :-2, :]
    Lt = L[2:, :, :] - L[:-2, :, :]
    
    T = min(Lx.shape[0], Ly.shape[0], Lt.shape[0])
    H = min(Lx.shape[1], Ly.shape[1], Lt.shape[1])
    W = min(Lx.shape[2], Ly.shape[2], Lt.shape[2])

    return Lx[:T, :H, :W], Ly[:T, :H, :W], Lt[:T, :H, :W]


# -------------------------------------------------------------
# Second Moment Matrix
# -------------------------------------------------------------

def second_moment_matrix(Lx, Ly, Lt, sigma=2.0, tau=1.5):
    """
    Compute smoothed second-moment matrix components.
    """
    J_xx = Lx * Lx
    J_xy = Lx * Ly
    J_xt = Lx * Lt
    J_yy = Ly * Ly
    J_yt = Ly * Lt
    J_tt = Lt * Lt

    def smooth(x):
        return gaussian_blur_3d(x, sigma, tau)

    return smooth(J_xx), smooth(J_xy), smooth(J_xt), smooth(J_yy), smooth(J_yt), smooth(J_tt)


# -------------------------------------------------------------
# Harris Interest Point Detection
# -------------------------------------------------------------

def harris_response(J, k=0.005):
    """
    Compute 3D Harris response.

    Args:
        J: second moment matrix tuple
        k: Harris constant
    """
    J_xx, J_xy, J_xt, J_yy, J_yt, J_tt = J

    det = (
        J_xx * (J_yy * J_tt - J_yt ** 2)
        - J_xy * (J_xy * J_tt - J_xt * J_yt)
        + J_xt * (J_xy * J_yt - J_xt * J_yy)
    )

    trace = J_xx + J_yy + J_tt
    return det - k * trace ** 3


def detect_interest_points(H, threshold_ratio=0.01):
    """
    Return coordinates of interest points.
    """
    threshold = threshold_ratio * H.max()
    points = torch.nonzero(H > threshold)
    return points


def extract_jet(L, x, y, t):
    """
    Compute first and second derivative jet at given coordinates.
    """
    Lc = L[t, y, x]

    # first order
    Lx  = L[t, y, x+1] - L[t, y, x-1]
    Ly  = L[t, y+1, x] - L[t, y-1, x]
    Lt  = L[t+1, y, x] - L[t-1, y, x]

    # second order
    Lxx = L[t, y, x+1] - 2*Lc + L[t, y, x-1]
    Lyy = L[t, y+1, x] - 2*Lc + L[t, y-1, x]
    Ltt = L[t+1, y, x] - 2*Lc + L[t-1, y, x]

    return Lx, Ly, Lt, Lxx, Lyy, Ltt


def extract_points(video, sigma, tau):
    """
    Extract interested points from a video.
    """
    L = gaussian_blur_3d(video, sigma, tau)
    Lx, Ly, Lt = gradients_3d(L)
    J = second_moment_matrix(Lx, Ly, Lt, 2*sigma, 2*tau)
    H = harris_response(J)
    points = detect_interest_points(H)

    return points

def extract_descriptors(video, sigma=1.5, tau=1.0):
    """
    Extract spatio-temporal jet descriptors from a video.

    Returns:
        Tensor shape (N,6)
    """
    L = gaussian_blur_3d(video, sigma, tau)
    Lx, Ly, Lt = gradients_3d(L)
    J = second_moment_matrix(Lx, Ly, Lt, 2*sigma, 2*tau)
    H = harris_response(J)
    t, y, x = detect_interest_points(H).T

    mask = (
        (t > 1) & (y > 1) & (x > 1) &
        (t < L.shape[0] - 2) &
        (y < L.shape[1] - 2) &
        (x < L.shape[2] - 2)
    )
    t, y, x = t[mask], y[mask], x[mask]
    if t.numel() == 0:
        return torch.empty((0, 6), device=L.device)
    
    sigma2 = sigma * sigma
    tau2 = tau * tau
    Lx, Ly, Lt, Lxx, Lyy, Ltt = extract_jet(L, x, y, t)
    desc = torch.stack([sigma * Lx, sigma * Ly, tau * Lt, sigma2 * Lxx, sigma2 * Lyy, tau2 * Ltt], dim=1)

    return desc
