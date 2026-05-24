import numpy as np
from torch.nn import functional as F
import torch
try:
    from kornia.geometry.conversions import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
    _USE_KORNIA = True
except ImportError:
    # Fallback when kornia is not available. ``angle_axis_to_rotation_matrix``
    # from torchgeometry is fine, but torchgeometry 0.1.2's
    # ``rotation_matrix_to_quaternion`` inverts boolean masks with
    # ``1 - mask``, which raises
    #   "Subtraction, the `-` operator, with a bool tensor is not supported"
    # on modern PyTorch. We use a dtype-safe local implementation of
    # ``rotation_matrix_to_angle_axis`` instead (assigned below).
    from torchgeometry import angle_axis_to_rotation_matrix
    _USE_KORNIA = False


def _rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """dtype-safe port of ``torchgeometry.rotation_matrix_to_quaternion``.

    Identical math to torchgeometry 0.1.2, but every boolean selection mask
    is cast to the matrix dtype before any arithmetic, so the ``1 - mask``
    inversions never touch a bool tensor (which modern PyTorch rejects).

    Shape: input ``(N, 3, 4)`` -> output ``(N, 4)``.
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    # Cast masks to the matrix dtype up front so all subsequent
    # arithmetic (including ``1 - mask``) stays in floating point.
    mask_d2 = (rmat_t[:, 2, 2] < eps).type_as(rotation_matrix)
    mask_d0_d1 = (rmat_t[:, 0, 0] > rmat_t[:, 1, 1]).type_as(rotation_matrix)
    mask_d0_nd1 = (rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]).type_as(rotation_matrix)

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def _quaternion_to_angle_axis(quaternion):
    """Port of ``torchgeometry.quaternion_to_angle_axis`` (no bool ops).

    Shape: input ``(*, 4)`` -> output ``(*, 3)``.
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def _rotation_matrix_to_angle_axis(rotation_matrix):
    """dtype-safe replacement for ``torchgeometry.rotation_matrix_to_angle_axis``."""
    quaternion = _rotation_matrix_to_quaternion(rotation_matrix)
    return _quaternion_to_angle_axis(quaternion)


if not _USE_KORNIA:
    rotation_matrix_to_angle_axis = _rotation_matrix_to_angle_axis

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_axis_angle(x, number = 36):
    ### input [B, J, 3,3 ]
    ### output [B, J*3]
    ###https://github.com/nkolot/SPIN/blob/master/train/trainer.py#L180
    ## may need to check https://github.com/kornia/kornia/pull/1270
    batch_size = x.shape[0]
    x2 = x.view(-1,3,3)
    # Convert predicted rotation matrices to axis-angle
    pred_rotmat_hom = torch.cat(
        [x2,
         torch.tensor([0, 0, 1], dtype=torch.float32,device=x.device).view(1, 3, 1).expand(batch_size * number, -1, -1)], dim=-1)
    pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
    # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
    pred_pose[torch.isnan(pred_pose)] = 0.0
    return pred_pose


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    if focal_length.shape[1] == 2:
        K[:,0,0] = focal_length[:,0]
        K[:,1,1] = focal_length[:,1]
    else:
        K[:, 0, 0] = focal_length[:].squeeze(1)
        K[:, 1, 1] = focal_length[:].squeeze(1)
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation.expand(batch_size,-1,-1), points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def rotmat_to_rot6d(x):
    #a1 = x[:, :, 0]
    #a2 = x[:, :, 1]
    # first one
    # return torch.cat([a1, a2], dim=-1)
    # second one
    #print(x[:, :, :2])
    return x[:, :, :2].reshape(-1 ,6)

# from https://github.com/huawei-noah/noah-research/tree/master/CLIFF
def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

# from PARE/geometry.py
def batch_euler2matrix(r):
    return quat_to_rotmat(euler_to_quaternion(r))

# from PARE/geometry.py
def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion

def rotation_angles_scipy_torch(matrix, degrees = True, order = 'zyx', extrinsic = True):
    '''
    rewrite from scipy/spatial/transform/rotation.py
    calculate the euler angle based on rotation matrix
    Args:
        matrix: [N,3,3]
        degrees: True or False
        seq: 'zyx' # only works for 'zyx
    Returns:
        angles: [N,3] #zyx#
    '''
    device = matrix.device
    if extrinsic:
        seq = order[::-1]
    pi = torch.Tensor([3.14159265358979323846]).type(torch.float32).to(device)

    num_rotations = matrix.shape[0]

    n1 = torch.Tensor([1.,0.,0.]).to(device) # x
    n2 = torch.Tensor([0.,1.,0.]).to(device) # y
    n3 = torch.Tensor([0.,0.,1.]).to(device) # z

    # step 2
    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    sl = torch.matmul(torch.cross(n1, n2, dim=0), n3)
    cl = torch.matmul(n1, n3)

    offset = torch.atan2(sl, cl)
    # c:[[0,1,0], [0,0,1],[1,0,0]]
    c = torch.vstack([n2, torch.cross(n1, n2), n1])

    # Step 3
    rot = torch.Tensor([[1, 0, 0],[0, cl, sl],[0, -sl, cl],]).to(device)
    res = torch.einsum('ij,bjk->bik', c, matrix)
    matrix_transformed = torch.einsum('bjk,ki->bji', res, c.T.matmul(rot))

    # Step 4
    angles = torch.ones([num_rotations, 3]).to(device)
    # Ensure less than unit norm
    positive_unity = matrix_transformed[:, 2, 2] > 1
    negative_unity = matrix_transformed[:, 2, 2] < -1
    matrix_transformed[positive_unity, 2, 2] = 1
    matrix_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = torch.arccos(matrix_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = (torch.abs(angles[:, 1]) >= eps)
    safe2 = (torch.abs(angles[:, 1] - pi) >= eps)

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = torch.logical_and(safe1, safe2)
    angles[safe_mask, 0] = torch.atan2(matrix_transformed[safe_mask, 0, 2],
                                      -matrix_transformed[safe_mask, 1, 2])
    angles[safe_mask, 2] = torch.atan2(matrix_transformed[safe_mask, 2, 0],
                                      matrix_transformed[safe_mask, 2, 1])
    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = torch.atan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 2] = -torch.atan2(matrix_transformed[~safe2, 1, 0]
                                        + matrix_transformed[~safe2, 0, 1],
                                        matrix_transformed[~safe2, 0, 0]
                                        - matrix_transformed[~safe2, 1, 1])
    else:
        # not used
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = torch.atan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 0] = torch.atan2(matrix_transformed[~safe2, 1, 0]
                                       + matrix_transformed[~safe2, 0, 1],
                                       matrix_transformed[~safe2, 0, 0]
                                       - matrix_transformed[~safe2, 1, 1])

    # Step 7
    if seq[0] == seq[2]:
        # not used
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = torch.logical_or(angles[:, 1] < 0, angles[:, 1] > pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = torch.logical_or(angles[:, 1] < -pi / 2,
                                    angles[:, 1] > pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = torch.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= pi

    angles[angles < -pi] += 2 * pi
    angles[angles > pi] -= 2 * pi

    # Reverse role of extrinsic and intrinsic rotations, but let third angle be
    # zero for gimbal locked cases
    if extrinsic:
        angles_final = torch.flip(angles, (1,))
    if degrees:
        angles_final = torch.rad2deg(angles_final)

    return angles_final