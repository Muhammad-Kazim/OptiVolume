import torch
import torch.nn as nn
import torch.nn.functional as F


### Quaternion Conversions

def from_axis_angle(axis, angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Tensor of shape (..., 3), where the magnitude is
            the rotation angle in radians around the vector's direction.

    Returns:
        Tensor of shape (..., 4): quaternions with real part first.
    """

    axis = axis / torch.norm(axis, p=2, dim=-1, keepdim=True)  # Normalize the axis

    axis_angle = axis * angle.unsqueeze(-1)  # Convert to axis-angle representation
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # (..., 1)
    half_angles = 0.5 * angles

    # Handle small angles robustly
    # sinc(x) = sin(pi*x)/(pi*x), so torch.sinc(half_angles/pi) = sin(half_angles)/half_angles
    sin_half_over_angle = torch.sinc(half_angles / torch.pi) * 0.5

    quats = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1
    )
    return quats

def to_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape (..., 4), where the last dimension
            represents the quaternion (w, x, y, z).
    Returns:
        Tensor of shape (..., 3, 3): rotation matrices.
    """

    norm = torch.linalg.vector_norm(
        quaternions, dim=-1, keepdim=True
    )

    quaternions = quaternions / norm.clamp_min(1e-8)

    w, x, y, z = torch.unbind(quaternions, dim=-1)

    two_s = 2.0 / (
        (quaternions * quaternions)
        .sum(-1, keepdim=True)
        .clamp_min(1e-8)
    )

    o = torch.stack([
        1 - two_s * (y*y + z*z),
        two_s * (x*y - z*w),
        two_s * (x*z + y*w),

        two_s * (x*y + z*w),
        1 - two_s * (x*x + z*z),
        two_s * (y*z - x*w),

        two_s * (x*z - y*w),
        two_s * (y*z + x*w),
        1 - two_s * (x*x + y*y),
    ], dim=-1)

    return o.reshape(quaternions.shape[:-1] + (3,3))

def to_axis_angle(quaternions):
    """
    Convert quaternions to axis/angle representation.

    Args:
        quaternions: Tensor of shape (..., 4), where the last dimension
            represents the quaternion (w, x, y, z).

    Returns:
        Tensor of shape (..., 3): axis-angle representation, where the
            magnitude is the rotation angle in radians around the vector's direction.
    """
    quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    
    w = quaternions[..., :1]           # (..., 1)
    v = quaternions[..., 1:]           # (..., 3)
    v_norm = v.norm(p=2, dim=-1, keepdim=True)  # (..., 1)

    # Compute full rotation angle
    angles = 2.0 * torch.atan2(v_norm, w)

    # Avoid division by zero (small angles)
    small = v_norm < 1e-8
    scale = torch.where(
        small,
        torch.ones_like(v_norm),          # arbitrary axis for tiny angles
        angles / v_norm
    )

    axis_angle = v * scale

    angles = torch.norm(axis_angle, p=2, dim=-1)  # (...,)
    axes = axis_angle / angles.unsqueeze(-1).clamp_min(1e-8)  # (..., 3)

    angles = torch.rad2deg(angles)  # Convert to degrees for better interpretability
    return axes, angles

def fmt_list(x, decimals=3):
    """
    Format a list of numbers for display, rounding to a specified number of decimal places.
    """
    return [round(num, decimals) for num in x]