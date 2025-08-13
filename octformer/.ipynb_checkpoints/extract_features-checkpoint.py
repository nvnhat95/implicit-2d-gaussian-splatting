#!/usr/bin/env python3
# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Feature Extraction Script (Simplified for NDFP + Segmentation Backbone)
# --------------------------------------------------------

import torch
import ocnn
import argparse
from typing import Dict, Tuple

import models


class FeatureExtractor:
    """Extract backbone features from OctFormerSeg given NDFP inputs."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """Initialize with segmentation model only.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run the model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load the segmentation model
        self.model = self._load_segmentation_model(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        print(f"Model loaded successfully from {checkpoint_path}")
        print("Model type: segmentation (backbone)")
        print(f"Device: {self.device}")

    def _load_segmentation_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Create OctFormerSeg and load weights."""
        # Create a segmentation model matching ScanNet200 configuration
        model = models.OctFormerSeg(
            in_channels=10,  # NDFP features (Normal 3 + Displacement 1 + Color 3 + Position 3)
            out_channels=201,  # Unused for backbone extraction, but required by the head
            channels=[96, 192, 384, 384],
            num_blocks=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 24],
            patch_size=32,
            dilation=4,
            drop_path=0.5,
            nempty=True,
            stem_down=2,
            head_up=2,
            fpn_channel=168,
            head_drop=[0.5, 0.5]
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model

    def extract_backbone_features(self, ndfp_points: torch.Tensor, octree_depth: int = 8) -> Dict[int, torch.Tensor]:
        """Extract multi-scale backbone features from NDFP points.

        Args:
            ndfp_points: Tensor (N, 10) in NDFP format
            octree_depth: Octree depth

        Returns:
            Dict mapping depth -> feature tensor
        """
        if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
            raise ValueError(f"ndfp_points must be (N, 10); got {tuple(ndfp_points.shape)}")

        with torch.no_grad():
            ndfp_points = ndfp_points.to(self.device)

            # Build octree from NDFP (use XYZ in channels 7:10 as coordinates, full 10-D as features)
            octree = self._build_octree_from_ndfp(ndfp_points, octree_depth)

            # Input features for backbone are the per-node 'F' features (10-D NDFP)
            input_feature = ocnn.modules.InputFeature('F', nempty=True)
            data = input_feature(octree)

            # Use segmentation backbone directly
            features = self.model.backbone(data, octree, octree.depth)
            return features

    def extract_features(self, ndfp_points: torch.Tensor, octree_depth: int = 8, position_encoding: str = 'displacement') -> Dict[str, Dict[int, torch.Tensor]]:
        """Extract both backbone (voxel) features and per-point relative position features.

        The relative position is computed from each gaussian (point) to the center of the
        voxel it belongs to at each depth.

        Args:
            ndfp_points: Tensor (N, 10) in NDFP format
            octree_depth: Octree depth
            position_encoding: 'displacement' | 'spherical' | 'both'

        Returns:
            Dictionary with keys:
            - 'voxel_features': Dict[depth, Tensor]
            - 'relative_positions': Dict[depth, Tensor]
        """
        if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
            raise ValueError(f"ndfp_points must be (N, 10); got {tuple(ndfp_points.shape)}")

        with torch.no_grad():
            ndfp_points = ndfp_points.to(self.device)

            # Build octree once
            octree = self._build_octree_from_ndfp(ndfp_points, octree_depth)

            # Backbone voxel features (per non-empty voxel)
            input_feature = ocnn.modules.InputFeature('F', nempty=True)
            data = input_feature(octree)
            voxel_features = self.model.backbone(data, octree, octree.depth)

            # Relative position features (per input point)
            relative_positions = self._compute_relative_positions(
                ndfp_points, octree, octree_depth, encoding_type=position_encoding
            )

            return {
                'voxel_features': voxel_features,
                'relative_positions': relative_positions,
            }

    def _build_octree_from_ndfp(self, ndfp_points: torch.Tensor, depth: int) -> ocnn.octree.Octree:
        """Build an octree using positions from NDFP and store NDFP as features."""
        coords = ndfp_points[:, 7:10]
        features = ndfp_points  # 10-D features

        # Create Points object
        point_cloud = ocnn.octree.Points(coords, features=features)

        # Build octree
        octree = ocnn.octree.Octree(depth, device=ndfp_points.device)
        octree.build_octree(point_cloud)
        octree.construct_all_neigh()
        return octree

    def _get_voxel_centers_and_mapping(self, ndfp_points: torch.Tensor, octree: ocnn.octree.Octree, depth: int) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Compute voxel centers and map each point to its voxel index at each depth."""
        voxel_centers: Dict[int, torch.Tensor] = {}
        point_to_voxel_mapping: Dict[int, torch.Tensor] = {}

        point_positions = ndfp_points[:, 7:10]

        for d in range(octree.full_depth, depth + 1):
            try:
                keys = octree.key(d, True)
            except Exception:
                keys = octree.key(d)

            if len(keys) == 0:
                continue

            x, y, z, _ = ocnn.octree.key2xyz(keys, d)
            
            centers = torch.stack([x, y, z], dim=1).to(torch.float32)
            voxel_centers[d] = centers

            # Map every point to its nearest voxel center
            # O(N * V) per depth; acceptable for typical usage
            distances = torch.cdist(point_positions, centers)
            closest_voxel = torch.argmin(distances, dim=1)
            point_to_voxel_mapping[d] = closest_voxel

        return voxel_centers, point_to_voxel_mapping

    def _compute_relative_positions(self, ndfp_points: torch.Tensor, octree: ocnn.octree.Octree, depth: int, encoding_type: str = 'displacement') -> Dict[int, torch.Tensor]:
        """Compute relative position features for each point to its voxel center.

        Supports three encodings:
        - displacement: vector from center to point (3)
        - spherical: distance, azimuth, elevation (3)
        - both: concatenation of displacement and spherical (6)
        """
        voxel_centers, point_to_voxel_mapping = self._get_voxel_centers_and_mapping(ndfp_points, octree, depth)
        point_positions = ndfp_points[:, 7:10]

        relative_positions: Dict[int, torch.Tensor] = {}
        for d in range(octree.full_depth, depth + 1):
            if d not in voxel_centers:
                continue

            centers = voxel_centers[d]
            mapping = point_to_voxel_mapping[d]

            # Gather voxel centers for each point
            selected_centers = centers[mapping]
            displacement = point_positions - selected_centers

            if encoding_type == 'displacement':
                relative_positions[d] = displacement
            elif encoding_type == 'spherical':
                distance = torch.norm(displacement, dim=1, keepdim=True)
                unit_vec = torch.where(distance > 1e-6, displacement / distance, torch.zeros_like(displacement))
                azimuth = torch.atan2(unit_vec[:, 1], unit_vec[:, 0]).unsqueeze(1)
                eps = torch.finfo(unit_vec.dtype).eps
                r_xy = torch.clamp(torch.sqrt(unit_vec[:, 0] ** 2 + unit_vec[:, 1] ** 2), min=eps).unsqueeze(1)
                elevation = torch.atan2(unit_vec[:, 2:3], r_xy)
                relative_positions[d] = torch.cat([distance, azimuth, elevation], dim=1)
            elif encoding_type == 'both':
                distance = torch.norm(displacement, dim=1, keepdim=True)
                unit_vec = torch.where(distance > 1e-6, displacement / distance, torch.zeros_like(displacement))
                azimuth = torch.atan2(unit_vec[:, 1], unit_vec[:, 0]).unsqueeze(1)
                eps = torch.finfo(unit_vec.dtype).eps
                r_xy = torch.clamp(torch.sqrt(unit_vec[:, 0] ** 2 + unit_vec[:, 1] ** 2), min=eps).unsqueeze(1)
                elevation = torch.atan2(unit_vec[:, 2:3], r_xy)
                spherical = torch.cat([distance, azimuth, elevation], dim=1)
                relative_positions[d] = torch.cat([displacement, spherical], dim=1)
            else:
                raise ValueError("position_encoding must be one of {'displacement', 'spherical', 'both'}")

        return relative_positions

    def get_feature_dimensions(self) -> Dict[int, int]:
        """Utility: run a dummy NDFP set to report backbone feature dims per depth."""
        dummy = torch.randn(2048, 10, device=self.device)
        features = self.extract_backbone_features(dummy)
        return {depth: feat.shape[1] for depth, feat in features.items()}


def main():
    parser = argparse.ArgumentParser(description='Extract OctFormerSeg backbone features from NDFP')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--ndfp', type=str, required=True, help='Path to a torch tensor file containing NDFP (N,10)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--octree_depth', type=int, default=8, help='Octree depth')
    args = parser.parse_args()

    # Initialize extractor
    extractor = FeatureExtractor(checkpoint_path=args.checkpoint, device=args.device)

    # Load NDFP points
    ndfp_points = torch.load(args.ndfp)
    if not isinstance(ndfp_points, torch.Tensor):
        raise TypeError('Loaded NDFP is not a torch.Tensor')
    if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
        raise ValueError(f'Expected NDFP tensor of shape (N, 10), got {tuple(ndfp_points.shape)}')

    # Extract features
    print(f"Extracting backbone features from {ndfp_points.shape[0]} points (NDFP)")
    features = extractor.extract_backbone_features(ndfp_points, octree_depth=args.octree_depth)

    # Report shapes
    print("\nFeature shapes per depth:")
    for depth, feat in features.items():
        print(f"  Depth {depth}: {tuple(feat.shape)}")

    # Optional utility: dimensions
    dims = extractor.get_feature_dimensions()
    print("\nBackbone feature dims per depth:")
    for depth, dim in dims.items():
        print(f"  Depth {depth}: {dim}")

    print("\nFeatures extracted successfully.")


if __name__ == '__main__':
    main() 