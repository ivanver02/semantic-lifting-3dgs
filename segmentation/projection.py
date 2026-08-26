import torch
from scene.cameras import Camera
from utils.graphics_utils import fov2focal, geom_transform_points

class GaussianProjector:
    """
    Handles the projection of 3D Gaussians into 2D camera space
    """
    def __init__(self, camera: Camera):
        """
        camera: the target camera for projection

        The projection uses the calibrated principal point reported by COLMAP,
        so the camera must expose the cx and cy attributes.
        """
        self.camera = camera
        missing = [name for name in ("cx", "cy") if not hasattr(camera, name)]
        if missing:
            raise AttributeError(
                f"camera {getattr(camera, 'image_name', repr(camera))!r} is missing "
                f"calibrated principal point attribute(s): {', '.join(missing)}"
            )
        self.device = camera.data_device
        self.width = camera.image_width
        self.height = camera.image_height

        # Calculate focal lengths from FoV
        self.focal_x = fov2focal(camera.FoVx, self.width)
        self.focal_y = fov2focal(camera.FoVy, self.height)

        # Store the world to view transform
        # The camera keeps it transposed, in the row vector convention of the rasterizer
        self.view_matrix = camera.world_view_transform

        # Rotational part W of the world to camera transform
        # Transposing the stored block undoes the row vector convention
        self.W = self.view_matrix[:3, :3].transpose(0, 1).contiguous()

    def project(self, means3D: torch.Tensor, cov3D: torch.Tensor):
        """
        Project 3D Gaussians to 2D
        The center of a Gaussian distribution is the mean, that is why the 3D coordinates of the center of the Gaussian are passed as means3D.
        The Gaussian covariance matrix is passed as cov3D
        """
        # Transform points to camera space
        means3D_cam = geom_transform_points(means3D, self.view_matrix)
        x, y, z = means3D_cam[:, 0], means3D_cam[:, 1], means3D_cam[:, 2]

        # Remove Gaussian centers behind the camera plane
        znear = self.camera.znear
        indices = torch.nonzero(z > znear, as_tuple=True)[0]
        x, y, z = x[indices], y[indices], z[indices]
        cov3D = cov3D[indices]

        '''
        Project Covariance
        Implements Sigma' = J W Sigma W^T J^T (Eq. 5)
        '''

        # Transform covariance into camera space
        w_matrix = self.W.unsqueeze(0).expand(cov3D.shape[0], -1, -1)
        w_sigma_wt = torch.bmm(w_matrix, torch.bmm(cov3D, w_matrix.transpose(1, 2)))

        '''
        J is the Jacobian of the affine approximation of the projective transformation, pi(x, y, z) = (f_x * x / z, f_y * y / z):

            [ fx/z   0   -(fx*x)/(z*z) ]
        J = [  0    fy/z -(fy*y)/(z*z) ] (Zwicker et al, formula 34)
            [  0     0         0       ]
        '''

        inv_z = 1.0 / z
        inv_z2 = inv_z * inv_z

        J = torch.zeros((x.shape[0], 2, 3), device=self.device)
        J[:, 0, 0] = self.focal_x * inv_z
        J[:, 0, 2] = -self.focal_x * x * inv_z2
        J[:, 1, 1] = self.focal_y * inv_z
        J[:, 1, 2] = -self.focal_y * y * inv_z2

        # Project covariance through the Jacobian
        cov2D = torch.bmm(J, torch.bmm(w_sigma_wt, J.transpose(1, 2)))

        # Add the isotropic low-pass filter used by the rasterizer, in pixel squared units
        cov2D[:, 0, 0] += 0.3
        cov2D[:, 1, 1] += 0.3

        # Apply the calibrated perspective projection with the principal point from COLMAP
        means2D = torch.stack([
            (x * self.focal_x * inv_z) + self.camera.cx,
            (y * self.focal_y * inv_z) + self.camera.cy,
        ], dim=1)

        return {
            'means2D': means2D,
            'cov2D': cov2D,
            'depths': z,
            'indices': indices,
        }
