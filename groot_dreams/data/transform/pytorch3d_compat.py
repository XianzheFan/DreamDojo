"""
Pure PyTorch implementations of pytorch3d.transforms rotation conversion functions.
Replaces the pytorch3d dependency for rotation representation conversions.
"""
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Converts 6D rotation representation to rotation matrix via Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrix to 6D rotation representation."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def _sign_not_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x).masked_fill_(x == 0, 1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to rotation matrices."""
    norm = torch.norm(quaternions, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    quaternions = quaternions / norm
    w, x, y, z = torch.unbind(quaternions, dim=-1)

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    one = torch.ones_like(x)

    matrix = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy),
    ], dim=-1)
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))

    x0, x1, x2, x3 = torch.unbind(q_abs, dim=-1)
    q = torch.stack([
        x0,
        x1 * _sign_not_zero(m21 - m12),
        x2 * _sign_not_zero(m02 - m20),
        x3 * _sign_not_zero(m10 - m01),
    ], dim=-1)
    return F.normalize(q, p=2, dim=-1)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to quaternions (w, x, y, z)."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    small_angles = angles.abs() < 1e-6
    sin_half_over_angles = torch.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        torch.sin(half_angles) / angles,
    )
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angles], dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to axis-angle."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    small_angles = angles.abs() < 1e-6
    sin_half_over_angles = torch.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        torch.sin(half_angles) / angles,
    )
    return quaternions[..., 1:] / sin_half_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrices."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def _index_from_letter(letter: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[letter]


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"axis must be X, Y, or Z, got {axis}")
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    return torch.atan2(-data[..., i1], data[..., i2])


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert Euler angles (radians) to rotation matrices."""
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert rotation matrices to Euler angles (radians)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0].clamp(-1.0, 1.0))

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)
