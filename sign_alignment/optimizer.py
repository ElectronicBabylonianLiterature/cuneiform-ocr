"""
Elastic Chain Optimizer for sign alignment refinement.

Uses gradient-based optimization to align text-based sign positions
with detection heatmap evidence.

Key insight: detection bounding-box *locations* are often more reliable
than their *class labels*. A class-agnostic "existence map" captures
the spatial probability that *any* sign exists at a position,
allowing the optimizer to snap signs to detected locations even when
the per-class label disagrees.

The data loss is therefore split into two parts:
    L_data = (1 - alpha_geo) * L_semantic  +  alpha_geo * L_geometric
where
    L_semantic  = -mean_i ScoreMap[class_i](x_i, y_i)   (per-class)
    L_geometric = -mean_i H_agnostic(x_i, y_i)          (class-agnostic)

Additionally, an IoU-based shape regression loss encourages the optimised
bounding boxes to match the *shape* (aspect ratio and area) of detector
outputs for the same class. For every class c that appears in both the
text-aligned signs and the detection list the loss is:
    L_iou = 1 - mean_c  GlobalIoU_c
where GlobalIoU_c is the IoU between the *union* of all optimised boxes
of class c and the *union* of all detected boxes of class c.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

from .sign import CLASSES_ABZ as DEFAULT_CLASSES_ABZ
from .tablet import SignBox, SubTablet
from .bounding_box import BoundingBox, Detection


def build_agnostic_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Build a class-agnostic existence map by taking the channel-wise max.

    Args:
        heatmap: (H, W, C) numpy array – per-class heatmap.

    Returns:
        (H, W) numpy array where each pixel is the max across all classes.
    """
    return np.max(heatmap, axis=2)


class ElasticChainOptimizer:
    """
    Elastic Chain Model for refining text-aligned bounding boxes.
    
    Uses gradient-based optimization to minimize an energy function that combines:
    - L_data: Heatmap matching score, decomposed into
        - L_semantic:  per-class heatmap response  (weight: 1-alpha_geo)
        - L_geometric: class-agnostic existence map (weight: alpha_geo)
    - L_iou: Per-class global IoU between optimised and detected boxes
    - L_seq: Sequential constraint (signs in a row should be tightly distributed)
    - L_smooth: Height consistency (adjacent signs should have similar heights)
    - L_anchor: Line baseline constraint (signs shouldn't deviate too far from row baseline)
    """
    
    def __init__(self, 
                 sub_tablet_text: SubTablet,
                 detection_heatmap: np.ndarray,
                 detection_boxes: Optional[Detection] = None,
                 classes_abz: List[str] = None,
                 scale_factor: int = 10,
                 lambda_data: float = 1.0,
                 lambda_iou: float = 0.0,
                 lambda_seq: float = 0.1,
                 lambda_smooth: float = 0.05,
                 lambda_anchor: float = 0.02,
                 alpha_geo: float = 0.7,
                 prior_aspect_ratio: float = 1.15,
                 device: str = None):
        """
        Args:
            sub_tablet_text: SubTablet with text-aligned sign boxes to optimize
            detection_heatmap: Heatmap from detection results (H, W, num_classes)
            detection_boxes: List of BoundingBox from detection (for IoU loss)
            classes_abz: List of ABZ class names (defaults to CLASSES_ABZ)
            scale_factor: Scale factor used in heatmap
            lambda_data: Weight for data term (heatmap matching)
            lambda_iou: Weight for IoU-based shape regression loss
            lambda_seq: Weight for sequential constraint
            lambda_smooth: Weight for height smoothness
            lambda_anchor: Weight for baseline anchor
            alpha_geo: Geometric (class-agnostic) weight inside L_data.
                       L_data = (1-alpha_geo)*L_semantic + alpha_geo*L_geometric.
                       Default 0.7 gives heavy weight to existence evidence.
            prior_aspect_ratio: Prior width/height ratio for signs
            device: Torch device ('cuda' or 'cpu')
        """
        self.sub_tablet_text = sub_tablet_text
        self.classes_abz = classes_abz if classes_abz is not None else DEFAULT_CLASSES_ABZ
        self.scale_factor = scale_factor
        self.lambda_data = lambda_data
        self.lambda_iou = lambda_iou
        self.lambda_seq = lambda_seq
        self.lambda_smooth = lambda_smooth
        self.lambda_anchor = lambda_anchor
        self.alpha_geo = alpha_geo
        self.prior_aspect_ratio = prior_aspect_ratio
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Convert per-class heatmap to torch tensor  (H, W, C)
        self.heatmap = torch.from_numpy(detection_heatmap).float().to(self.device)
        
        # Build and store class-agnostic existence map  (H, W)
        agnostic_np = build_agnostic_heatmap(detection_heatmap)
        self.heatmap_agnostic = torch.from_numpy(agnostic_np).float().to(self.device)
        
        # Get rows of sign boxes
        self.rows = sub_tablet_text.get_rows()
        self.num_rows = len(self.rows)
        
        # Build flat list and index mapping
        self.sign_boxes_flat: List[SignBox] = []
        self.row_indices = []
        self.col_indices = []
        self.class_ids = []
        
        for row_idx, row in enumerate(self.rows):
            for col_idx, sb in enumerate(row):
                self.sign_boxes_flat.append(sb)
                self.row_indices.append(row_idx)
                self.col_indices.append(col_idx)
                if sb.abz_name in self.classes_abz:
                    self.class_ids.append(self.classes_abz.index(sb.abz_name))
                else:
                    self.class_ids.append(-1)
        
        self.num_signs = len(self.sign_boxes_flat)
        
        # Initialize optimization parameters: [cx, cy, w, h] for each sign
        init_params = [[sb.cx, sb.cy, sb.width, sb.height] for sb in self.sign_boxes_flat]
        self.params = torch.tensor(init_params, dtype=torch.float32,
                                   device=self.device, requires_grad=True)
        self.initial_params = self.params.clone().detach()
        
        # Compute initial row baselines
        self.row_baselines = []
        for row in self.rows:
            if row:
                avg_y = np.mean([sb.cy for sb in row])
                self.row_baselines.append(avg_y)
            else:
                self.row_baselines.append(0)
        self.row_baselines = torch.tensor(self.row_baselines, dtype=torch.float32,
                                          device=self.device)
        
        # ---- Detection boxes for IoU loss ----
        # Group detected boxes by class_id (ABZ index)
        self.det_boxes_by_class: Dict[int, torch.Tensor] = {}
        if detection_boxes is not None and self.lambda_iou > 0:
            from collections import defaultdict
            tmp: Dict[int, list] = defaultdict(list)
            for bb in detection_boxes:
                abz = bb.sign.abz
                if abz in self.classes_abz:
                    cid = self.classes_abz.index(abz)
                    tmp[cid].append([bb.x1, bb.y1, bb.x2, bb.y2])
            for cid, box_list in tmp.items():
                self.det_boxes_by_class[cid] = torch.tensor(
                    box_list, dtype=torch.float32, device=self.device
                )  # (N_det_c, 4)

        self.loss_history = []
        self.loss_components_history = []
    
    def _sample_heatmap_at_positions(self, heatmap_2d: torch.Tensor) -> torch.Tensor:
        """
        Bilinear-sample a 2-D map (H, W) at current sign center positions.
        Returns a 1-D tensor of sampled values (one per sign).
        """
        heatmap_h, heatmap_w = heatmap_2d.shape
        # Prepare (1, 1, H, W) for grid_sample
        map_4d = heatmap_2d.unsqueeze(0).unsqueeze(0)

        scores = []
        for i in range(self.num_signs):
            cx = self.params[i, 0] / self.scale_factor
            cy = self.params[i, 1] / self.scale_factor

            norm_x = (cx / (heatmap_w - 1)) * 2 - 1
            norm_y = (cy / (heatmap_h - 1)) * 2 - 1
            norm_x = torch.clamp(norm_x, -1, 1)
            norm_y = torch.clamp(norm_y, -1, 1)

            grid = torch.stack([norm_x, norm_y]).view(1, 1, 1, 2)
            val = F.grid_sample(map_4d, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
            scores.append(val.squeeze())

        return torch.stack(scores)

    # ---- Semantic loss (per-class) ----
    def compute_semantic_loss(self) -> torch.Tensor:
        """L_semantic = -mean_i ScoreMap[class_i](x_i, y_i)."""
        loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        heatmap_h, heatmap_w, num_classes = self.heatmap.shape
        
        for i in range(self.num_signs):
            class_id = self.class_ids[i]
            if class_id < 0 or class_id >= num_classes:
                continue
            
            cx = self.params[i, 0] / self.scale_factor
            cy = self.params[i, 1] / self.scale_factor
            
            norm_x = (cx / (heatmap_w - 1)) * 2 - 1
            norm_y = (cy / (heatmap_h - 1)) * 2 - 1
            norm_x = torch.clamp(norm_x, -1, 1)
            norm_y = torch.clamp(norm_y, -1, 1)
            
            grid = torch.stack([norm_x, norm_y]).view(1, 1, 1, 2)
            single_heatmap = self.heatmap[:, :, class_id].unsqueeze(0).unsqueeze(0)
            score = F.grid_sample(single_heatmap, grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
            
            loss = loss - score.squeeze()
            valid_count += 1
        
        return loss / max(1, valid_count)

    # ---- Geometric loss (class-agnostic) ----
    def compute_geometric_loss(self) -> torch.Tensor:
        """L_geometric = -mean_i H_agnostic(x_i, y_i)."""
        scores = self._sample_heatmap_at_positions(self.heatmap_agnostic)
        return -scores.mean()

    # ---- Combined data loss ----
    def compute_data_loss(self) -> torch.Tensor:
        """
        L_data = (1 - alpha_geo) * L_semantic + alpha_geo * L_geometric

        When alpha_geo = 0  → pure per-class matching (old behaviour).
        When alpha_geo = 1  → pure existence-map matching.
        Default alpha_geo = 0.7 heavily relies on geometric evidence.
        """
        L_sem = self.compute_semantic_loss()
        L_geo = self.compute_geometric_loss()
        return (1.0 - self.alpha_geo) * L_sem + self.alpha_geo * L_geo

    # ---- IoU-based shape regression loss ----
    def compute_iou_loss(self) -> torch.Tensor:
        """
        Per-class global IoU loss to encourage shape diversity.

        For each class c present in both text-aligned signs and detections:
          1. Collect all optimised boxes of class c → compute their bounding
             union (min x1, min y1, max x2, max y2) and sum of areas.
          2. Collect all detected  boxes of class c (fixed targets).
          3. Compute the IoU of the two *aggregate bounding boxes*.
          4. L_iou_c = 1 - IoU_c.

        Returns the mean (1 - IoU) across all matched classes.
        """
        if not self.det_boxes_by_class:
            return torch.tensor(0.0, device=self.device)

        # Group optimised sign indices by class
        opt_by_class: Dict[int, List[int]] = {}
        for i in range(self.num_signs):
            cid = self.class_ids[i]
            if cid < 0:
                continue
            if cid not in self.det_boxes_by_class:
                continue
            opt_by_class.setdefault(cid, []).append(i)

        if not opt_by_class:
            return torch.tensor(0.0, device=self.device)

        iou_losses = []
        for cid, indices in opt_by_class.items():
            # Optimised boxes for this class: derive (x1, y1, x2, y2)
            cx  = torch.stack([self.params[i, 0] for i in indices])
            cy  = torch.stack([self.params[i, 1] for i in indices])
            w   = torch.stack([self.params[i, 2] for i in indices])
            h   = torch.stack([self.params[i, 3] for i in indices])
            ox1 = cx - w / 2
            oy1 = cy - h / 2
            ox2 = cx + w / 2
            oy2 = cy + h / 2

            # Aggregate bounding union of optimised boxes
            opt_min_x1 = ox1.min()
            opt_min_y1 = oy1.min()
            opt_max_x2 = ox2.max()
            opt_max_y2 = oy2.max()

            # Detected boxes for this class (fixed target)
            det_boxes = self.det_boxes_by_class[cid]  # (N, 4)
            det_min_x1 = det_boxes[:, 0].min()
            det_min_y1 = det_boxes[:, 1].min()
            det_max_x2 = det_boxes[:, 2].max()
            det_max_y2 = det_boxes[:, 3].max()

            # IoU between the two aggregate boxes
            inter_x1 = torch.max(opt_min_x1, det_min_x1)
            inter_y1 = torch.max(opt_min_y1, det_min_y1)
            inter_x2 = torch.min(opt_max_x2, det_max_x2)
            inter_y2 = torch.min(opt_max_y2, det_max_y2)

            inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
            inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
            inter_area = inter_w * inter_h

            opt_area = (opt_max_x2 - opt_min_x1) * (opt_max_y2 - opt_min_y1)
            det_area = (det_max_x2 - det_min_x1) * (det_max_y2 - det_min_y1)
            union_area = opt_area + det_area - inter_area

            iou = inter_area / (union_area + 1e-6)
            iou_losses.append(1.0 - iou)

        return torch.stack(iou_losses).mean()
    
    def compute_seq_loss(self) -> torch.Tensor:
        """L_seq: adjacent signs in a row should be tightly distributed."""
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        idx = 0
        for row in self.rows:
            row_len = len(row)
            for j in range(row_len - 1):
                cx_j = self.params[idx + j, 0]
                cx_j1 = self.params[idx + j + 1, 0]
                w_j = self.params[idx + j, 2]
                w_j1 = self.params[idx + j + 1, 2]
                
                expected_gap = (w_j + w_j1) / 2
                actual_gap = cx_j1 - cx_j
                loss = loss + (actual_gap - expected_gap) ** 2
                count += 1
            idx += row_len
        
        return loss / max(1, count)
    
    def compute_smooth_loss(self) -> torch.Tensor:
        """Height consistency + aspect ratio prior."""
        loss_height = torch.tensor(0.0, device=self.device)
        loss_aspect = torch.tensor(0.0, device=self.device)
        count_height = 0
        count_aspect = 0
        
        idx = 0
        for row in self.rows:
            row_len = len(row)
            for j in range(row_len):
                w_j = self.params[idx + j, 2]
                h_j = self.params[idx + j, 3]
                
                loss_aspect = loss_aspect + (w_j - self.prior_aspect_ratio * h_j) ** 2
                count_aspect += 1
                
                if j < row_len - 1:
                    h_j1 = self.params[idx + j + 1, 3]
                    loss_height = loss_height + (h_j - h_j1) ** 2
                    count_height += 1
            idx += row_len
        
        return loss_height / max(1, count_height) + loss_aspect / max(1, count_aspect)
    
    def compute_anchor_loss(self) -> torch.Tensor:
        """Signs should stay near their row baseline."""
        loss = torch.tensor(0.0, device=self.device)
        
        idx = 0
        for row_idx, row in enumerate(self.rows):
            baseline = self.row_baselines[row_idx]
            row_len = len(row)
            for j in range(row_len):
                cy = self.params[idx + j, 1]
                loss = loss + (cy - baseline) ** 2
            idx += row_len
        
        return loss / max(1, self.num_signs)
    
    def compute_total_loss(self) -> tuple:
        """Compute total loss and return all components."""
        L_sem = self.compute_semantic_loss()
        L_geo = self.compute_geometric_loss()
        L_data = (1.0 - self.alpha_geo) * L_sem + self.alpha_geo * L_geo
        L_iou = self.compute_iou_loss()
        L_seq = self.compute_seq_loss()
        L_smooth = self.compute_smooth_loss()
        L_anchor = self.compute_anchor_loss()
        
        L_total = (self.lambda_data * L_data +
                   self.lambda_iou * L_iou +
                   self.lambda_seq * L_seq +
                   self.lambda_smooth * L_smooth +
                   self.lambda_anchor * L_anchor)
        
        return L_total, L_data, L_iou, L_seq, L_smooth, L_anchor, L_sem, L_geo
    
    def optimize(self, num_iterations: int = 100, lr: float = 1.0,
                 verbose: bool = True, log_every: int = 10) -> SubTablet:
        """
        Run optimization and return optimized SubTablet.
        
        Args:
            num_iterations: Number of optimization iterations
            lr: Learning rate
            verbose: Print progress
            log_every: Print every N iterations
            
        Returns:
            Optimized SubTablet
        """
        optimizer = torch.optim.Adam([self.params], lr=lr)
        
        if verbose:
            print(f"Starting optimization with {self.num_signs} signs, {self.num_rows} rows")
            print(f"Lambdas: data={self.lambda_data}, iou={self.lambda_iou}, seq={self.lambda_seq}, "
                  f"smooth={self.lambda_smooth}, anchor={self.lambda_anchor}")
            print(f"alpha_geo={self.alpha_geo} (data = {1-self.alpha_geo:.2f}*semantic + {self.alpha_geo:.2f}*geometric)")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            L_total, L_data, L_iou, L_seq, L_smooth, L_anchor, L_sem, L_geo = self.compute_total_loss()
            L_total.backward()
            optimizer.step()
            
            # Ensure positive widths and heights
            with torch.no_grad():
                self.params[:, 2] = torch.clamp(self.params[:, 2], min=10)
                self.params[:, 3] = torch.clamp(self.params[:, 3], min=10)
            
            self.loss_history.append(L_total.item())
            self.loss_components_history.append({
                'total': L_total.item(),
                'data': L_data.item(),
                'iou': L_iou.item(),
                'semantic': L_sem.item(),
                'geometric': L_geo.item(),
                'seq': L_seq.item(),
                'smooth': L_smooth.item(),
                'anchor': L_anchor.item()
            })
            
            if verbose and (iteration % log_every == 0 or iteration == num_iterations - 1):
                print(f"Iter {iteration:4d}: L_total={L_total.item():.4f}, "
                      f"L_data={L_data.item():.4f} (sem={L_sem.item():.4f}, geo={L_geo.item():.4f}), "
                      f"L_iou={L_iou.item():.4f}, "
                      f"L_seq={L_seq.item():.4f}, L_smooth={L_smooth.item():.4f}, "
                      f"L_anchor={L_anchor.item():.4f}")
        
        return self.get_optimized_subtablet()
    
    def get_optimized_subtablet(self) -> SubTablet:
        """Create a new SubTablet with optimized parameters."""
        optimized_params = self.params.detach().cpu().numpy()
        
        new_sign_boxes = []
        for i, sb in enumerate(self.sign_boxes_flat):
            new_sb = SignBox(
                sign=sb.sign,
                score=sb.score,
                cx=float(optimized_params[i, 0]),
                cy=float(optimized_params[i, 1]),
                width=float(optimized_params[i, 2]),
                height=float(optimized_params[i, 3]),
                row_idx=sb.row_idx,
                col_idx=sb.col_idx
            )
            new_sign_boxes.append(new_sb)
        
        return SubTablet(
            img=self.sub_tablet_text.img,
            sign_boxes=new_sign_boxes,
            heatmap=None,
            name="optimized",
            scale_factor=self.sub_tablet_text.scale_factor,
            avg_width=self.sub_tablet_text.avg_width,
            avg_height=self.sub_tablet_text.avg_height,
            margin=self.sub_tablet_text.margin,
            origin_x=self.sub_tablet_text.origin_x,
            origin_y=self.sub_tablet_text.origin_y
        )
    
    def get_param_changes(self) -> np.ndarray:
        """Get param changes from initial to current."""
        current = self.params.detach().cpu().numpy()
        initial = self.initial_params.cpu().numpy()
        return current - initial
    
    def plot_loss_history(self, figsize: tuple = (20, 4)):
        """Plot loss history."""
        if not self.loss_components_history:
            print("No optimization history available")
            return
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # 1. Total loss
        axes[0].plot(self.loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss over Iterations')
        axes[0].grid(True)
        
        # 2. Main components
        for comp in ['data', 'iou', 'seq', 'smooth', 'anchor']:
            values = [h.get(comp, 0) for h in self.loss_components_history]
            axes[1].plot(values, label=f'L_{comp}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components over Iterations')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. Semantic vs Geometric breakdown
        sem_vals = [h.get('semantic', h.get('data', 0)) for h in self.loss_components_history]
        geo_vals = [h.get('geometric', 0) for h in self.loss_components_history]
        axes[2].plot(sem_vals, label='L_semantic (per-class)')
        axes[2].plot(geo_vals, label='L_geometric (agnostic)')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title(f'Data Loss Decomposition (α_geo={self.alpha_geo:.2f})')
        axes[2].legend()
        axes[2].grid(True)
        
        # 4. IoU loss
        iou_vals = [h.get('iou', 0) for h in self.loss_components_history]
        axes[3].plot(iou_vals, label='L_iou', color='tab:red')
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('1 - IoU')
        axes[3].set_title(f'IoU Shape Regression Loss (λ_iou={self.lambda_iou})')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
