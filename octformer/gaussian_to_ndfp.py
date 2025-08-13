import torch
from typing import Optional


def build_rotation(q: torch.Tensor) -> torch.Tensor:
	"""Convert batched quaternions [x, y, z, w] to rotation matrices (N, 3, 3).
	Ensures normalization and preserves dtype/device.
	"""
	if q.dim() != 2 or q.shape[-1] != 4:
		raise ValueError("build_rotation expects shape (N, 4) quaternion input")
	# Normalize quaternion
	eps = torch.finfo(q.dtype).eps if q.is_floating_point() else 1e-8
	norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
	qx, qy, qz, qw = (q / norm).unbind(-1)

	xx = qx * qx
	yy = qy * qy
	zz = qz * qz
	xy = qx * qy
	xz = qx * qz
	yz = qy * qz
	wx = qw * qx
	wy = qw * qy
	wz = qw * qz

	m00 = 1 - 2 * (yy + zz)
	m01 = 2 * (xy - wz)
	m02 = 2 * (xz + wy)
	m10 = 2 * (xy + wz)
	m11 = 1 - 2 * (xx + zz)
	m12 = 2 * (yz - wx)
	m20 = 2 * (xz - wy)
	m21 = 2 * (yz + wx)
	m22 = 1 - 2 * (xx + yy)

	R = torch.stack([
		torch.stack([m00, m01, m02], dim=-1),
		torch.stack([m10, m11, m12], dim=-1),
		torch.stack([m20, m21, m22], dim=-1),
	], dim=-2)
	return R


class GaussianToNDFP:
	"""Convert Gaussian Splatting points into NDFP format.

	NDFP format per point: [Normal(3), Displacement(1), Color RGB(3), Position XYZ(3)] -> 10 values.

	- Normal: computed from Gaussian rotation by rotating the canonical z-axis.
	- Displacement: constant or provided tensor (default 0.0).
	- Color: taken from DC SH coefficients using sigmoid.
	- Position: Gaussian xyz.
	"""

	def __init__(self, displacement_value: float = 0.0, device: Optional[torch.device] = None):
		self.displacement_value = float(displacement_value)
		self.device = device

	@staticmethod
	def _ensure_device(t: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
		if device is None or t.device == device:
			return t
		return t.to(device)

	@staticmethod
	def _dc_sh_to_rgb(dc_sh: torch.Tensor) -> torch.Tensor:
		"""Convert DC SH coefficients (degree-0) to RGB.
		Uses SH constant Y00 = 0.28209479177387814 so rgb = dc_sh * Y00.
		Expects shape (N, 3). Returns (N, 3) clamped to [0, 1].
		"""
		SH_C0 = 0.28209479177387814
		rgb = dc_sh * SH_C0
		return torch.clamp(rgb, 0.0, 1.0)

	@staticmethod
	def _rotation_to_normal(rotation_matrix: torch.Tensor) -> torch.Tensor:
		"""Compute normals by rotating the canonical z-axis with the given rotation matrices.
		rotation_matrix: (N, 3, 3)
		returns normals: (N, 3)
		"""
		canonical_z = torch.tensor([0.0, 0.0, 1.0], device=rotation_matrix.device, dtype=rotation_matrix.dtype)
		normals = torch.matmul(rotation_matrix, canonical_z)  # (N, 3)
		return normals

	def convert_from_model(self, model) -> torch.Tensor:
		"""Convert from a GaussianModel instance to an NDFP tensor of shape (N, 10).
		The model is expected to provide:
		  - model.get_xyz -> (N, 3)
		  - model._rotation -> (N, 4) quaternion compatible with build_rotation
		  - model._features_dc -> (N, 1, 3) DC SH per RGB channel
		"""
		xyz = model.get_xyz  # (N, 3)
		rotation_q = model._rotation  # (N, 4)
		features_dc = model._features_dc  # (N, 1, 3)

		# Ensure device consistency
		xyz = self._ensure_device(xyz, self.device) if self.device is not None else xyz
		rotation_q = self._ensure_device(rotation_q, xyz.device)
		features_dc = self._ensure_device(features_dc, xyz.device)

		# Build rotation matrices and normals
		rotation_matrix = build_rotation(rotation_q)  # (N, 3, 3)
		normals = self._rotation_to_normal(rotation_matrix)  # (N, 3)

		# Color from DC SH via sigmoid
		dc = features_dc.squeeze(1)  # (N, 3)
		rgb = self._dc_sh_to_rgb(dc)  # (N, 3)

		# Displacement
		displacement = torch.full((xyz.shape[0], 1), self.displacement_value, device=xyz.device, dtype=xyz.dtype)

		# Compose NDFP
		ndfp = torch.cat([normals, displacement, rgb, xyz], dim=1)  # (N, 10)
		return ndfp

	def convert_from_tensors(
		self,
		xyz: torch.Tensor,
		rotation: torch.Tensor,
		dc_sh_features: Optional[torch.Tensor] = None,
		rgb: Optional[torch.Tensor] = None,
		displacement: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Convert directly from tensors to NDFP.

		Args:
			xyz: (N, 3)
			rotation: (N, 4) quaternion compatible with build_rotation, or (N, 3, 3) rotation matrices
			dc_sh_features: (N, 3) DC SH coefficients for RGB channels; used if rgb is None
			rgb: (N, 3) direct RGB colors in [0, 1]; overrides dc_sh_features if provided
			displacement: (N, 1) per-point displacement; if None, uses constant value from init

		Returns:
			Tensor (N, 10) in NDFP format
		"""
		xyz = self._ensure_device(xyz, self.device) if self.device is not None else xyz

		# Rotation to normals
		if rotation.dim() == 2 and rotation.shape[-1] == 4:
			rotation = self._ensure_device(rotation, xyz.device)
			rotation_matrix = build_rotation(rotation)
		elif rotation.dim() == 3 and rotation.shape[-2:] == (3, 3):
			rotation_matrix = self._ensure_device(rotation, xyz.device)
		else:
			raise ValueError("rotation must be (N, 4) quaternion or (N, 3, 3) rotation matrix")
		normals = self._rotation_to_normal(rotation_matrix)

		# Color
		if rgb is None:
			if dc_sh_features is None:
				raise ValueError("Either rgb or dc_sh_features must be provided")
			dc_sh_features = self._ensure_device(dc_sh_features, xyz.device)
			rgb = self._dc_sh_to_rgb(dc_sh_features)
		else:
			rgb = self._ensure_device(rgb, xyz.device)

		# Displacement
		if displacement is None:
			displacement = torch.full((xyz.shape[0], 1), self.displacement_value, device=xyz.device, dtype=xyz.dtype)
		else:
			displacement = self._ensure_device(displacement, xyz.device)

		# Compose NDFP
		ndfp = torch.cat([normals, displacement, rgb, xyz], dim=1)
		return ndfp 