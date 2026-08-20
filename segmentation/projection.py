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
        """
        self.camera = camera
        missing = [name for name in ("cx", "cy") if not hasattr(camera, name)]
        if missing:
            camera_name = getattr(
                camera, "image_name", getattr(camera, "uid", repr(camera)),

    # Raise the missing calibration error
            )
            raise AttributeError(
                f"camera {camera_name!r} is missing calibrated principal point "
                f"attribute(s): {', '.join(missing)}"
            )
        self.device = camera.data_device
        self.width = camera.image_width
        self.height = camera.image_height

        # Calculate focal lengths from FoV
        self.focal_x = fov2focal(camera.FoVx, self.width)
        self.focal_y = fov2focal(camera.FoVy, self.height)
        
        # Store the world to view rotation block
        self.view_matrix = camera.world_view_transform
        
        # Extract the leading 3 by 3 rotation block for covariance rotation
        self.W = self.view_matrix[:3, :3] 

    def project(self, means3D: torch.Tensor, cov3D: torch.Tensor):
        """
        Project 3D Gaussians to 2D
        The center of a Gaussian distribution is the mean, that is why the 3D coordinates of the center of the Gaussian are passed as means3D. 
        The Gaussian covariance matrix is passed as cov3D
        """
        # Transform points to camera space
        means3D_cam = geom_transform_points(means3D, self.view_matrix)
        
        # Extract x, y, z from those points
        x, y, z = means3D_cam[:, 0], means3D_cam[:, 1], means3D_cam[:, 2]
        
        # Remove Gaussian centers behind the camera plane
        znear = self.camera.znear
        mask_z = z > znear
        
        # Applying the mask
        # Keep a one dimensional index tensor for visible Gaussians
        indices = torch.nonzero(mask_z, as_tuple=True)[0]
        x = x[indices]
        y = y[indices]
        z = z[indices]
        cov3D = cov3D[indices]
        
        '''
        Project Covariance
        Implements Sigma' = J W Sigma W^T J^T (Eq. 5)
        '''
        
        # Transform covariance into camera space
        w_matrix = self.W 
        w_sigma_wt = torch.bmm(w_matrix.T.unsqueeze(0).expand(cov3D.shape[0], -1, -1), 
                              torch.bmm(cov3D, w_matrix.unsqueeze(0).expand(cov3D.shape[0], -1, -1)))
        
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

        # Add the isotropic filter used by the rasterizer
        # The covariance uses pixel squared units
        cov2D[:, 0, 0] += 0.3
        cov2D[:, 1, 1] += 0.3
        
        # Compute projected means to 2D:
        # Apply the calibrated perspective projection
        # Use the calibrated principal point
        cx = self.camera.cx
        cy = self.camera.cy
        
        means2D = torch.stack([
            (x * self.focal_x * inv_z) + cx,
            (y * self.focal_y * inv_z) + cy
        ], dim=1)
        
        return {
            'means2D': means2D,
            'cov2D': cov2D,
            'depths': z,
            'indices': indices
        }