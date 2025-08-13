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

    def extract_features(self, ndfp_points: torch.Tensor, octree_depth: int = 8, position_encoding: str = 'displacement', octree: ocnn.octree.Octree = None, batch_size: int = 10000, use_cpu_fallback: bool = True) -> Dict[str, Dict[int, torch.Tensor]]:
        """Extract both backbone (voxel) features and per-point relative position features.

        The relative position is computed from each gaussian (point) to the center of the
        voxel it belongs to at each depth.

        Args:
            ndfp_points: Tensor (N, 10) in NDFP format
            octree_depth: Octree depth (ignored if octree is provided)
            position_encoding: 'displacement' | 'spherical' | 'both'
            octree: Pre-built octree (optional). If None, builds octree from ndfp_points.
            batch_size: Number of points to process at once to avoid OOM
            use_cpu_fallback: If True, fallback to CPU computation on OOM

        Returns:
            Dictionary with keys:
            - 'voxel_features': Dict[depth, Tensor]
            - 'relative_positions': Dict[depth, Tensor]
        """
        if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
            raise ValueError(f"ndfp_points must be (N, 10); got {tuple(ndfp_points.shape)}")

        with torch.no_grad():
            ndfp_points = ndfp_points.to(self.device)

            # Build octree if not provided
            if octree is None:
                octree = self._build_octree_from_ndfp(ndfp_points, octree_depth)

            # Backbone voxel features (per non-empty voxel)
            input_feature = ocnn.modules.InputFeature('F', nempty=True)
            data = input_feature(octree)
            voxel_features = self.model.backbone(data, octree, octree.depth)

            # Relative position features (per input point)
            relative_positions = self._compute_relative_positions(
                ndfp_points, octree, octree.depth, encoding_type=position_encoding,
                batch_size=batch_size, use_cpu_fallback=use_cpu_fallback
            )

            return {
                'voxel_features': voxel_features,
                'relative_positions': relative_positions,
            }

    def extract_features_from_octree(self, ndfp_points: torch.Tensor, octree: ocnn.octree.Octree, position_encoding: str = 'displacement', batch_size: int = 10000, use_cpu_fallback: bool = True) -> Dict[str, Dict[int, torch.Tensor]]:
        """Extract features from a pre-built octree.
        
        This method is designed for batch processing where the octree is built once
        with all points and then features are extracted for batches of points.

        Args:
            ndfp_points: Tensor (N, 10) in NDFP format (subset of points used to build octree)
            octree: Pre-built octree from all points
            position_encoding: 'displacement' | 'spherical' | 'both'
            batch_size: Number of points to process at once to avoid OOM
            use_cpu_fallback: If True, fallback to CPU computation on OOM

        Returns:
            Dictionary with keys:
            - 'voxel_features': Dict[depth, Tensor] - full voxel features from octree
            - 'relative_positions': Dict[depth, Tensor] - only for the input ndfp_points
        """
        if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
            raise ValueError(f"ndfp_points must be (N, 10); got {tuple(ndfp_points.shape)}")

        with torch.no_grad():
            ndfp_points = ndfp_points.to(self.device)

            # Backbone voxel features (per non-empty voxel) - computed from full octree
            input_feature = ocnn.modules.InputFeature('F', nempty=True)
            data = input_feature(octree)
            voxel_features = self.model.backbone(data, octree, octree.depth)

            # Relative position features only for the subset of points
            relative_positions = self._compute_relative_positions(
                ndfp_points, octree, octree.depth, encoding_type=position_encoding,
                batch_size=batch_size, use_cpu_fallback=use_cpu_fallback
            )

            return {
                'voxel_features': voxel_features,
                'relative_positions': relative_positions,
            }

    def build_octree(self, ndfp_points: torch.Tensor, octree_depth: int = 8) -> ocnn.octree.Octree:
        """Build octree from NDFP points. 
        
        This is a public method to allow building octree once before batch processing.
        
        Args:
            ndfp_points: Tensor (N, 10) in NDFP format
            octree_depth: Octree depth
            
        Returns:
            Built octree
        """
        if ndfp_points.dim() != 2 or ndfp_points.shape[1] != 10:
            raise ValueError(f"ndfp_points must be (N, 10); got {tuple(ndfp_points.shape)}")
            
        ndfp_points = ndfp_points.to(self.device)
        return self._build_octree_from_ndfp(ndfp_points, octree_depth)

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

    def _get_voxel_centers_and_mapping(self, ndfp_points: torch.Tensor, octree: ocnn.octree.Octree, depth: int, batch_size: int = 10000, use_cpu_fallback: bool = True) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Compute voxel centers and map each point to its voxel index at each depth.
        
        Args:
            ndfp_points: Input points tensor
            octree: Octree structure
            depth: Maximum depth to process
            batch_size: Number of points to process at once to avoid OOM
            use_cpu_fallback: If True, fallback to CPU computation on OOM
        """
        voxel_centers: Dict[int, torch.Tensor] = {}
        point_to_voxel_mapping: Dict[int, torch.Tensor] = {}

        point_positions = ndfp_points[:, 7:10]
        num_points = point_positions.shape[0]

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

            # Memory-efficient closest voxel assignment
            torch.cuda.empty_cache()  # Clear GPU cache
            
            try:
                # Try batched computation on GPU
                closest_voxel = self._compute_closest_voxels_batched(point_positions, centers, batch_size)
            except RuntimeError as e:
                if ("out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower()) and use_cpu_fallback:
                    print(f"GPU batched OOM at depth {d}. Falling back to CPU computation...")
                    torch.cuda.empty_cache()
                    # Move to CPU for computation
                    closest_voxel = self._compute_closest_voxels_cpu(point_positions, centers, batch_size)
                else:
                    raise e
            
            point_to_voxel_mapping[d] = closest_voxel

        return voxel_centers, point_to_voxel_mapping

    def _compute_closest_voxels(self, point_positions: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute closest voxels using full distance matrix (fastest but most memory intensive)."""
        distances = torch.cdist(point_positions, centers)
        return torch.argmin(distances, dim=1)

    def _compute_closest_voxels_batched(self, point_positions: torch.Tensor, centers: torch.Tensor, batch_size: int = 10000) -> torch.Tensor:
        """Compute closest voxels using batched approach to reduce memory usage."""
        num_points = point_positions.shape[0]
        closest_voxels = torch.empty(num_points, dtype=torch.long, device=point_positions.device)
        
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_points = point_positions[start_idx:end_idx]
            
            # Compute distances for this batch
            batch_distances = torch.cdist(batch_points, centers)
            batch_closest = torch.argmin(batch_distances, dim=1)
            
            closest_voxels[start_idx:end_idx] = batch_closest
            
        return closest_voxels

    def _compute_closest_voxels_cpu(self, point_positions: torch.Tensor, centers: torch.Tensor, batch_size: int = 10000) -> torch.Tensor:
        """Compute closest voxels on CPU with batching (slowest but most memory efficient)."""
        # Move to CPU
        point_positions_cpu = point_positions.cpu()
        centers_cpu = centers.cpu()
        
        num_points = point_positions_cpu.shape[0]
        closest_voxels = torch.empty(num_points, dtype=torch.long)
        
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_points = point_positions_cpu[start_idx:end_idx]
            
            # Compute distances for this batch on CPU
            batch_distances = torch.cdist(batch_points, centers_cpu)
            batch_closest = torch.argmin(batch_distances, dim=1)
            
            closest_voxels[start_idx:end_idx] = batch_closest
            
        # Move result back to original device
        return closest_voxels.to(point_positions.device)

    def _compute_relative_positions(self, ndfp_points: torch.Tensor, octree: ocnn.octree.Octree, depth: int, encoding_type: str = 'displacement', batch_size: int = 10000, use_cpu_fallback: bool = True) -> Dict[int, torch.Tensor]:
        """Compute relative position features for each point to its voxel center.

        Supports three encodings:
        - displacement: vector from center to point (3)
        - spherical: distance, azimuth, elevation (3)
        - both: concatenation of displacement and spherical (6)
        
        Args:
            ndfp_points: Input points tensor
            octree: Octree structure  
            depth: Maximum depth to process
            encoding_type: Type of position encoding
            batch_size: Number of points to process at once to avoid OOM
            use_cpu_fallback: If True, fallback to CPU computation on OOM
        """
        voxel_centers, point_to_voxel_mapping = self._get_voxel_centers_and_mapping(
            ndfp_points, octree, depth, batch_size=batch_size, use_cpu_fallback=use_cpu_fallback
        )
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