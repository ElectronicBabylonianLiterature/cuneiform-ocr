"""
Elastic Chain Optimizer for sign alignment refinement.

Uses gradient-based optimization to align text-based sign positions
with detection heatmap evidence.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional

from .sign import CLASSES_ABZ as DEFAULT_CLASSES_ABZ
from .tablet import SignBox, SubTablet


class ElasticChainOptimizer:
    """
    Elastic Chain Model for refining text-aligned bounding boxes.
    
    Uses gradient-based optimization to minimize an energy function that combines:
    - L_data: Heatmap matching score (signs should be at high-response positions)
    - L_seq: Sequential constraint (signs in a row should be tightly distributed)
    - L_smooth: Height consistency (adjacent signs should have similar heights)
    - L_anchor: Line baseline constraint (signs shouldn't deviate too far from row baseline)
    """
    
    def __init__(self, 
                 sub_tablet_text: SubTablet,
                 detection_heatmap: np.ndarray,
                 classes_abz: List[str] = None,
                 scale_factor: int = 10,
                 lambda_data: float = 1.0,
                 lambda_seq: float = 0.1,
                 lambda_smooth: float = 0.05,
                 lambda_anchor: float = 0.02,
                 prior_aspect_ratio: float = 1.15,
                 device: str = None):
        """
        Args:
            sub_tablet_text: SubTablet with text-aligned sign boxes to optimize
            detection_heatmap: Heatmap from detection results (H, W, num_classes)
            classes_abz: List of ABZ class names (defaults to CLASSES_ABZ)
            scale_factor: Scale factor used in heatmap
            lambda_data: Weight for data term (heatmap matching)
            lambda_seq: Weight for sequential constraint
            lambda_smooth: Weight for height smoothness
            lambda_anchor: Weight for baseline anchor
            prior_aspect_ratio: Prior width/height ratio for signs
            device: Torch device ('cuda' or 'cpu')
        """
        self.sub_tablet_text = sub_tablet_text
        self.classes_abz = classes_abz if classes_abz is not None else DEFAULT_CLASSES_ABZ
        self.scale_factor = scale_factor
        self.lambda_data = lambda_data
        self.lambda_seq = lambda_seq
        self.lambda_smooth = lambda_smooth
        self.lambda_anchor = lambda_anchor
        self.prior_aspect_ratio = prior_aspect_ratio
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Convert heatmap to torch tensor
        self.heatmap = torch.from_numpy(detection_heatmap).float().to(self.device)
        
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
        
        self.loss_history = []
        self.loss_components_history = []
    
    def compute_data_loss(self) -> torch.Tensor:
        """L_data = -sum_i ScoreMap[class_i](x_i, y_i), using grid_sample."""
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
        L_data = self.compute_data_loss()
        L_seq = self.compute_seq_loss()
        L_smooth = self.compute_smooth_loss()
        L_anchor = self.compute_anchor_loss()
        
        L_total = (self.lambda_data * L_data +
                   self.lambda_seq * L_seq +
                   self.lambda_smooth * L_smooth +
                   self.lambda_anchor * L_anchor)
        
        return L_total, L_data, L_seq, L_smooth, L_anchor
    
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
            print(f"Lambdas: data={self.lambda_data}, seq={self.lambda_seq}, "
                  f"smooth={self.lambda_smooth}, anchor={self.lambda_anchor}")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            L_total, L_data, L_seq, L_smooth, L_anchor = self.compute_total_loss()
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
                'seq': L_seq.item(),
                'smooth': L_smooth.item(),
                'anchor': L_anchor.item()
            })
            
            if verbose and (iteration % log_every == 0 or iteration == num_iterations - 1):
                print(f"Iter {iteration:4d}: L_total={L_total.item():.4f}, "
                      f"L_data={L_data.item():.4f}, L_seq={L_seq.item():.4f}, "
                      f"L_smooth={L_smooth.item():.4f}, L_anchor={L_anchor.item():.4f}")
        
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
    
    def plot_loss_history(self, figsize: tuple = (12, 4)):
        """Plot loss history."""
        if not self.loss_components_history:
            print("No optimization history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(self.loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss over Iterations')
        axes[0].grid(True)
        
        for comp in ['data', 'seq', 'smooth', 'anchor']:
            values = [h[comp] for h in self.loss_components_history]
            axes[1].plot(values, label=f'L_{comp}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components over Iterations')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
