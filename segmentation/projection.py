import torch
from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import fov2focal, geom_transform_points


def get_covariance_3d(gaussians) -> torch.Tensor:
    """
    Compute the full 3D covariance matrix for each Gaussian, Sigma = R S S^T R^T

    Returns:
        torch.Tensor: tensor containing covariance matrices, with shape (N, 3, 3)
    """

    # Build the scaled rotation matrix
    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)

    # Form the covariance matrix
    return L @ L.transpose(1, 2)  # Shape: (N, 3, 3)


def project_gaussians(camera, means3D: torch.Tensor, cov3D: torch.Tensor):
    """
    Project 3D Gaussians to 2D
    The center of a Gaussian distribution is the mean, that is why the 3D coordinates of the center of the Gaussian are passed as means3D.
    The Gaussian covariance matrix is passed as cov3D

    The projection uses the calibrated principal point reported by COLMAP (camera.cx and camera.cy)
    """

    # Calculate focal lengths from FoV
    focal_x = fov2focal(camera.FoVx, camera.image_width)
    focal_y = fov2focal(camera.FoVy, camera.image_height)

    # The camera keeps the world to view transform transposed, in the row vector convention of the rasterizer
    view_matrix = camera.world_view_transform

    # Transform points to camera space
    means3D_cam = geom_transform_points(means3D, view_matrix)
    x, y, z = means3D_cam[:, 0], means3D_cam[:, 1], means3D_cam[:, 2]

    # Remove Gaussian centers behind the camera plane
    indices = torch.nonzero(z > camera.znear, as_tuple=True)[0]
    x, y, z = x[indices], y[indices], z[indices]
    cov3D = cov3D[indices]

    '''
    Project Covariance
    Implements Sigma' = J W Sigma W^T J^T (Eq. 5)
    '''

    # Rotational part W of the world to camera transform
    # Transposing the stored block undoes the row vector convention
    W = view_matrix[:3, :3].transpose(0, 1).contiguous()

    # Transform covariance into camera space
    w_matrix = W.unsqueeze(0).expand(cov3D.shape[0], -1, -1)
    w_sigma_wt = torch.bmm(w_matrix, torch.bmm(cov3D, w_matrix.transpose(1, 2)))

    '''
    J is the Jacobian of the affine approximation of the projective transformation, pi(x, y, z) = (f_x * x / z, f_y * y / z):

        [ fx/z   0   -(fx*x)/(z*z) ]
    J = [  0    fy/z -(fy*y)/(z*z) ] (Zwicker et al, formula 34)
        [  0     0         0       ]
    '''

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z

    J = torch.zeros((x.shape[0], 2, 3), device=means3D.device)
    J[:, 0, 0] = focal_x * inv_z
    J[:, 0, 2] = -focal_x * x * inv_z2
    J[:, 1, 1] = focal_y * inv_z
    J[:, 1, 2] = -focal_y * y * inv_z2

    # Project covariance through the Jacobian
    cov2D = torch.bmm(J, torch.bmm(w_sigma_wt, J.transpose(1, 2)))

    # Add the isotropic low-pass filter used by the rasterizer, in pixel squared units
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3

    # Apply the calibrated perspective projection with the principal point from COLMAP
    means2D = torch.stack([
        (x * focal_x * inv_z) + camera.cx,
        (y * focal_y * inv_z) + camera.cy,
    ], dim=1)

    return {
        'means2D': means2D,
        'cov2D': cov2D,
        'depths': z,
        'indices': indices,
    }
