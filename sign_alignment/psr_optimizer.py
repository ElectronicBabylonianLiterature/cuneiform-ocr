"""
Point Set Registration (PSR) Optimizer for sign alignment.

Replaces heatmap-based template matching with a GMM-based point set
registration approach.  The source set S (text-derived signs, organized
in rows) is optimized to match the target set X (detected signs).

Data loss (GMM):
    E_data = -(1/N) ∑_n log p(x_n)
    p(x_n) = w/N + (1-w) · (1/M) · ∑_m  W_{c_m,c_n} · N(x_n; s_m, σ²I)

where W is a class confusion matrix (default: identity → only same-class
matches contribute).  The noise term w/N acts as a uniform background
that absorbs unmatched detections.

Structural losses preserve the row topology of S:
    L_anchor  — signs stay near their row baseline (fitted line)
    L_seq     — adjacent signs in a row are evenly spaced
    L_height  — heights within a row are consistent (low variance)
    L_rows    — inter-row spacing matches the average sign height
                (piecewise quadratic + plateau)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from .sign import CLASSES_ABZ as DEFAULT_CLASSES_ABZ
from .tablet import SignBox, SubTablet
from .bounding_box import BoundingBox, Detection


# ================================================================
#  Utility functions
# ================================================================

def initialize_text_subtablet(
    text_lines: List[List[str]],
    target_detections: Detection,
    avg_width: float,
    avg_height: float,
    margin: float = None,
    img: np.ndarray = None,
) -> SubTablet:
    """
    Create a text SubTablet whose centroid coincides with the detection
    centroid – a simple, heatmap-free initialisation.

    Steps
    -----
    1. Lay out text signs on a uniform grid (one row per text line).
    2. Compute the centroid of the grid.
    3. Compute the centroid of the target detections.
    4. Translate the whole grid so the two centroids coincide.

    Parameters
    ----------
    text_lines : list[list[str]]
        Parsed text lines (each inner list contains sign names).
    target_detections : Detection
        Bounding boxes from the object detector.
    avg_width, avg_height : float
        Average sign dimensions used for grid spacing.
    margin : float, optional
        Grid margin (defaults to ``max(avg_width, avg_height)``).
    img : np.ndarray, optional
        Image to attach to the SubTablet (for visualisation).

    Returns
    -------
    SubTablet
        With ``sign_boxes`` placed at the centroid-adjusted positions.
    """
    return SubTablet.from_text_lines(
        text_lines=text_lines,
        avg_width=avg_width,
        avg_height=avg_height,
        margin=margin,
        img=img,
        target_detections=target_detections,
        align_to_detection_centroid=True,
        name="text_initialized",
    )


def filter_boxes_by_mask(
    subtablet: SubTablet,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> SubTablet:
    """
    Remove sign boxes whose centres fall outside a binary mask.

    Parameters
    ----------
    subtablet : SubTablet
        Input subtablet.
    mask : np.ndarray
        Binary mask (H, W). Values ≥ threshold are considered "inside".
    threshold : float
        Threshold in [0, 1] if max(mask) ≤ 1, else in [0, 255].

    Returns
    -------
    SubTablet
        Copy with only the boxes whose centres are inside the mask.
    """
    mask_h, mask_w = mask.shape[:2]
    thr_val = threshold * 255 if mask.max() > 1 else threshold

    filtered: List[SignBox] = []
    for sb in subtablet.sign_boxes:
        px, py = int(round(sb.cx)), int(round(sb.cy))
        if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px] >= thr_val:
            filtered.append(sb.copy())

    return SubTablet(
        img=subtablet.img,
        sign_boxes=filtered,
        name=subtablet.name + "_masked",
        scale_factor=subtablet.scale_factor,
        avg_width=subtablet.avg_width,
        avg_height=subtablet.avg_height,
        margin=subtablet.margin,
        origin_x=subtablet.origin_x,
        origin_y=subtablet.origin_y,
    )


# ================================================================
#  Main optimizer
# ================================================================

class PointSetRegistrationOptimizer:
    """
    GMM-based Point Set Registration optimizer for cuneiform sign alignment.

    Source set S (M points in rows, from text) is optimised to match
    target set X (N points, from the detector).
    """

    def __init__(
        self,
        sub_tablet_text: SubTablet,
        target_detections: Detection,
        classes_abz: List[str] = None,
        sigma: float = None,
        w_noise: float = 0.1,
        confusion_matrix: np.ndarray = None,
        lambda_data: float = 1.0,
        lambda_anchor: float = 0.1,
        lambda_seq: float = 0.1,
        lambda_height: float = 0.1,
        lambda_rows: float = 0.1,
        rows_threshold_ratio: float = 1.0 / 3.0,
        device: str = None,
    ):
        """
        Parameters
        ----------
        sub_tablet_text : SubTablet
            Text-derived sign boxes (source S). Must have ``row_idx``
            and ``col_idx`` set on each ``SignBox``.
        target_detections : Detection
            Bounding boxes from the detector (target X).
        classes_abz : list[str]
            ABZ class name list (defaults to ``CLASSES_ABZ``).
        sigma : float
            GMM bandwidth.  Default: ``(avg_width + avg_height) / 2``.
        w_noise : float
            Noise weight in [0, 1).
        confusion_matrix : np.ndarray (C, C)
            Class matching weights.  Default: identity (same-class only).
        lambda_data .. lambda_rows : float
            Loss weights.
        rows_threshold_ratio : float
            Fraction of ideal inter-row spacing that defines the
            quadratic → plateau transition in ``L_rows``.
        device : str
            ``'cuda'`` or ``'cpu'``.
        """
        self.classes_abz = classes_abz or DEFAULT_CLASSES_ABZ
        self.w_noise = w_noise
        self.lambda_data = lambda_data
        self.lambda_anchor = lambda_anchor
        self.lambda_seq = lambda_seq
        self.lambda_height = lambda_height
        self.lambda_rows = lambda_rows
        self.rows_threshold_ratio = rows_threshold_ratio
        self.sub_tablet_text = sub_tablet_text

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # ---- Source point set S (text) -----------------------------------
        self.rows = sub_tablet_text.get_rows()
        self.num_rows = len(self.rows)
        self.row_lengths = [len(r) for r in self.rows]

        self.sign_boxes_flat: List[SignBox] = []
        self.row_indices: List[int] = []
        self.col_indices: List[int] = []
        self.class_ids_source: List[int] = []

        for row_idx, row in enumerate(self.rows):
            for col_idx, sb in enumerate(row):
                self.sign_boxes_flat.append(sb)
                self.row_indices.append(row_idx)
                self.col_indices.append(col_idx)
                cid = (self.classes_abz.index(sb.abz_name)
                       if sb.abz_name in self.classes_abz else -1)
                self.class_ids_source.append(cid)

        self.M = len(self.sign_boxes_flat)

        # ---- Target point set X (detections) -----------------------------
        self.target_detections = target_detections
        self.N = len(target_detections)

        target_pos = []
        target_sz = []
        self.class_ids_target: List[int] = []

        for det in target_detections:
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2
            target_pos.append([cx, cy])
            target_sz.append([det.width, det.height])
            cid = (self.classes_abz.index(det.sign.abz)
                   if det.sign.abz in self.classes_abz else -1)
            self.class_ids_target.append(cid)

        self.X_pos = (torch.tensor(target_pos, dtype=torch.float32, device=self.device)
                      if self.N > 0
                      else torch.zeros(0, 2, dtype=torch.float32, device=self.device))
        self.X_size = (torch.tensor(target_sz, dtype=torch.float32, device=self.device)
                       if self.N > 0
                       else torch.zeros(0, 2, dtype=torch.float32, device=self.device))

        self.class_ids_target_t = torch.tensor(self.class_ids_target, dtype=torch.long,
                                               device=self.device)
        self.class_ids_source_t = torch.tensor(self.class_ids_source, dtype=torch.long,
                                               device=self.device)

        # ---- Optimisation parameters [cx, cy, w, h] ---------------------
        init = [[sb.cx, sb.cy, sb.width, sb.height]
                for sb in self.sign_boxes_flat]
        self.params = torch.tensor(init, dtype=torch.float32,
                                   device=self.device, requires_grad=True)
        self.initial_params = self.params.clone().detach()

        # ---- Sigma -------------------------------------------------------
        avg_size = (sub_tablet_text.avg_width + sub_tablet_text.avg_height) / 2
        self.sigma = float(sigma) if sigma is not None else float(avg_size)

        # ---- Confusion matrix (C×C) -------------------------------------
        C = len(self.classes_abz)
        if confusion_matrix is not None:
            self.confusion_matrix = torch.tensor(
                confusion_matrix, dtype=torch.float32, device=self.device)
        else:
            self.confusion_matrix = torch.eye(C, dtype=torch.float32,
                                              device=self.device)

        # Pre-compute class weight matrix (M, N) --------------------------
        self._build_class_weight_matrix()

        # ---- Row offsets for flat indexing --------------------------------
        self.row_offsets: List[int] = []
        offset = 0
        for rl in self.row_lengths:
            self.row_offsets.append(offset)
            offset += rl

        # ---- History -----------------------------------------------------
        self.loss_history: List[float] = []
        self.loss_components_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _build_class_weight_matrix(self):
        """W[m, n] = confusion_matrix[c_m, c_n], 0 for unknown classes."""
        if self.M == 0 or self.N == 0:
            self.W_class = torch.zeros(self.M, self.N, device=self.device)
            return

        src_valid = self.class_ids_source_t >= 0  # (M,)
        tgt_valid = self.class_ids_target_t >= 0  # (N,)
        valid_mask = src_valid.unsqueeze(1) & tgt_valid.unsqueeze(0)  # (M, N)

        src_idx = self.class_ids_source_t.clamp(min=0)
        tgt_idx = self.class_ids_target_t.clamp(min=0)

        W = self.confusion_matrix[src_idx][:, tgt_idx]  # (M, N)
        self.W_class = W * valid_mask.float()

    # ==================================================================
    #  Loss functions
    # ==================================================================

    def compute_data_loss(self) -> torch.Tensor:
        """
        GMM data loss (numerically stable log-space computation).

        E_data = -(1/N) Σ_n log p(x_n)
        p(x_n) = w/N + (1-w)/M · Σ_m W[m,n] · N(x_n; s_m, σ²I)
        """
        if self.N == 0 or self.M == 0:
            return torch.tensor(0.0, device=self.device)

        S_pos = self.params[:, :2]  # (M, 2)

        # Pairwise squared distance  (M, N)
        diff = S_pos.unsqueeze(1) - self.X_pos.unsqueeze(0)  # (M, N, 2)
        dist_sq = (diff ** 2).sum(dim=2)                       # (M, N)

        # Log Gaussian:  log N(x|s,σ²I) = -D/2 log(2πσ²) - ||x-s||²/(2σ²)
        D = 2.0
        log_norm = -D / 2.0 * np.log(2.0 * np.pi * self.sigma ** 2)
        log_gauss = log_norm - dist_sq / (2.0 * self.sigma ** 2)  # (M, N)

        # log(W[m,n]) – use a large negative *finite* floor instead of
        # -inf so that logsumexp never sees an all-inf column (which
        # would produce NaN gradients via 0/0 in the internal softmax).
        _LOG_FLOOR = -1e8
        log_W = torch.full((self.M, self.N), _LOG_FLOOR, device=self.device)
        W_pos = self.W_class > 0
        if W_pos.any():
            log_W[W_pos] = torch.log(self.W_class[W_pos])

        # log weighted Gaussian per (m, n)
        log_weighted = log_W + log_gauss  # (M, N) ; _LOG_FLOOR where W=0

        # log mixture_n = log((1-w)/M) + logsumexp_m
        coeff = (1.0 - self.w_noise) / max(self.M, 1)
        log_coeff = np.log(max(coeff, 1e-30))
        log_mixture = torch.logsumexp(log_weighted, dim=0) + log_coeff  # (N,)

        # log p(x_n) = logaddexp(log_mixture, log_noise)
        noise_val = self.w_noise / max(self.N, 1)
        log_noise = np.log(max(noise_val, 1e-30))
        log_noise_t = torch.tensor(log_noise, dtype=torch.float32,
                                   device=self.device)
        log_p = torch.logaddexp(log_mixture, log_noise_t)  # (N,)

        return -log_p.mean()

    # ------------------------------------------------------------------

    def compute_anchor_loss(self) -> torch.Tensor:
        """
        Signs stay near their row baseline (fitted line).

        For each row with ≥ 2 signs, fit y = a·x + b, then return the
        mean squared deviation from the line.
        """
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            if rl < 2:
                continue

            x = self.params[off:off + rl, 0]
            y = self.params[off:off + rl, 1]

            x_mean, y_mean = x.mean(), y.mean()
            xc = x - x_mean
            denom = (xc ** 2).sum()

            if denom < 1e-6:
                dev = y - y_mean
            else:
                slope = (xc * (y - y_mean)).sum() / denom
                intercept = y_mean - slope * x_mean
                dev = y - (slope * x + intercept)

            loss = loss + (dev ** 2).mean()
            count += 1

        return loss / max(1, count)

    # ------------------------------------------------------------------

    def compute_seq_loss(self) -> torch.Tensor:
        """
        Adjacent signs in a row are evenly spaced.

        expected_gap = (w_j + w_{j+1}) / 2
        L_seq = mean (actual_gap − expected_gap)²
        """
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            for j in range(rl - 1):
                cx_j = self.params[off + j, 0]
                cx_j1 = self.params[off + j + 1, 0]
                w_j = self.params[off + j, 2]
                w_j1 = self.params[off + j + 1, 2]

                expected = (w_j + w_j1) / 2
                actual = cx_j1 - cx_j
                loss = loss + (actual - expected) ** 2
                count += 1

        return loss / max(1, count)

    # ------------------------------------------------------------------

    def compute_height_loss(self) -> torch.Tensor:
        """
        Heights within a row should be consistent  (low variance).
        """
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            if rl < 2:
                continue
            h = self.params[off:off + rl, 3]
            loss = loss + h.var()
            count += 1

        return loss / max(1, count)

    # ------------------------------------------------------------------

    def _fit_row_baseline(self, ri: int):
        """Fit y = slope·x + intercept to a row's sign centres.

        Returns (slope, intercept, x_min, x_max) as tensors.
        """
        off = self.row_offsets[ri]
        rl = self.row_lengths[ri]
        x = self.params[off:off + rl, 0]
        y = self.params[off:off + rl, 1]

        if rl <= 1:
            s = torch.tensor(0.0, device=self.device)
            b = y[0] if rl == 1 else torch.tensor(0.0, device=self.device)
            xmn = x[0] if rl == 1 else torch.tensor(0.0, device=self.device)
            xmx = xmn
        else:
            xm, ym = x.mean(), y.mean()
            xc = x - xm
            denom = (xc ** 2).sum().clamp(min=1e-6)
            s = (xc * (y - ym)).sum() / denom
            b = ym - s * xm
            xmn = x.min()
            xmx = x.max()
        return s, b, xmn, xmx

    def compute_rows_loss(self) -> torch.Tensor:
        """
        Inter-row spacing should match the ideal (average height).

        ideal_spacing = (mean_h_row_i + mean_h_row_{i+1}) / 2
        actual_spacing = baseline_{i+1}(x_align) − baseline_i(x_align)
        x_align = max(first_x_row_i, first_x_row_{i+1})

        Piecewise loss per pair:
            t = ideal · threshold_ratio           (default 1/3)
            d = actual − ideal
            |d| ≤ t :  L = (d/t)²
            |d| > t :  L = 1              (constant plateau)
        """
        if self.num_rows < 2:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for i in range(self.num_rows - 1):
            off_i = self.row_offsets[i]
            off_j = self.row_offsets[i + 1]
            rl_i = self.row_lengths[i]
            rl_j = self.row_lengths[i + 1]
            if rl_i == 0 or rl_j == 0:
                continue

            # Average heights of the two rows
            h_i = self.params[off_i:off_i + rl_i, 3].mean()
            h_j = self.params[off_j:off_j + rl_j, 3].mean()
            ideal = (h_i + h_j) / 2

            # Fitted baselines
            s_i, b_i, xmin_i, _ = self._fit_row_baseline(i)
            s_j, b_j, xmin_j, _ = self._fit_row_baseline(i + 1)

            # Evaluate baselines at aligned x (where both rows exist)
            x_align = torch.max(xmin_i, xmin_j)
            y_i = s_i * x_align + b_i
            y_j = s_j * x_align + b_j
            actual = y_j - y_i  # should be positive (top→bottom)

            deviation = actual - ideal
            threshold = (ideal * self.rows_threshold_ratio).clamp(min=1e-6)
            normalised = deviation / threshold

            # Piecewise: quadratic inside, constant outside
            loss_pair = torch.where(
                normalised.abs() <= 1.0,
                normalised ** 2,
                torch.ones_like(normalised),
            )
            loss = loss + loss_pair
            count += 1

        return loss / max(1, count)

    # ------------------------------------------------------------------
    #  Total loss
    # ------------------------------------------------------------------

    def compute_total_loss(self):
        L_data = self.compute_data_loss()
        L_anchor = self.compute_anchor_loss()
        L_seq = self.compute_seq_loss()
        L_height = self.compute_height_loss()
        L_rows = self.compute_rows_loss()

        L_total = (self.lambda_data * L_data
                   + self.lambda_anchor * L_anchor
                   + self.lambda_seq * L_seq
                   + self.lambda_height * L_height
                   + self.lambda_rows * L_rows)

        return L_total, L_data, L_anchor, L_seq, L_height, L_rows

    # ==================================================================
    #  Optimisation loop
    # ==================================================================

    def optimize(
        self,
        num_iterations: int = 200,
        lr: float = 1.0,
        sigma_anneal: bool = False,
        sigma_final: float = None,
        verbose: bool = True,
        log_every: int = 10,
    ) -> SubTablet:
        """
        Run gradient-descent optimisation.

        Parameters
        ----------
        num_iterations : int
        lr : float
            Learning rate for Adam.
        sigma_anneal : bool
            Whether to linearly anneal sigma from its initial value to
            ``sigma_final``.
        sigma_final : float
            Final sigma (default: ``sigma / 4``).
        verbose, log_every : int
            Printing control.

        Returns
        -------
        SubTablet
            Optimised sign positions.
        """
        opt = torch.optim.Adam([self.params], lr=lr)

        sigma_init = self.sigma
        if sigma_final is None:
            sigma_final = self.sigma / 4.0

        if verbose:
            print("=" * 60)
            print("Point Set Registration Optimizer")
            print("=" * 60)
            print(f"  Source (text):  {self.M} signs, {self.num_rows} rows")
            print(f"  Target (det):   {self.N} signs")
            print(f"  Device:         {self.device}")
            sigma_str = (f"{sigma_init:.1f} → {sigma_final:.1f}"
                         if sigma_anneal else f"{self.sigma:.1f}")
            print(f"  sigma:          {sigma_str}")
            print(f"  w_noise:        {self.w_noise}")
            print(f"  Lambdas:  data={self.lambda_data}, anchor={self.lambda_anchor}, "
                  f"seq={self.lambda_seq}, height={self.lambda_height}, "
                  f"rows={self.lambda_rows}")
            print("-" * 60)

        for it in range(num_iterations):
            # Optional sigma annealing
            if sigma_anneal:
                t = it / max(1, num_iterations - 1)
                self.sigma = sigma_init * (1.0 - t) + sigma_final * t

            opt.zero_grad()
            L_total, L_data, L_anchor, L_seq, L_height, L_rows = \
                self.compute_total_loss()
            L_total.backward()

            # Guard against NaN / exploding gradients
            if self.params.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.params], max_norm=1e4)
                if torch.isnan(self.params.grad).any():
                    self.params.grad.nan_to_num_(nan=0.0)

            opt.step()

            # Positive width/height
            with torch.no_grad():
                self.params[:, 2].clamp_(min=10.0)
                self.params[:, 3].clamp_(min=10.0)

            self.loss_history.append(L_total.item())
            self.loss_components_history.append({
                'total': L_total.item(),
                'data': L_data.item(),
                'anchor': L_anchor.item(),
                'seq': L_seq.item(),
                'height': L_height.item(),
                'rows': L_rows.item(),
                'sigma': self.sigma,
            })

            if verbose and (it % log_every == 0 or it == num_iterations - 1):
                line = (f"Iter {it:4d}: total={L_total.item():.4f}  "
                        f"data={L_data.item():.4f}  "
                        f"anchor={L_anchor.item():.4f}  "
                        f"seq={L_seq.item():.4f}  "
                        f"height={L_height.item():.4f}  "
                        f"rows={L_rows.item():.4f}")
                if sigma_anneal:
                    line += f"  σ={self.sigma:.1f}"
                print(line)

        return self.get_optimized_subtablet()

    # ------------------------------------------------------------------
    #  Results
    # ------------------------------------------------------------------

    def get_optimized_subtablet(self) -> SubTablet:
        """Build a new SubTablet from the current optimised parameters."""
        p = self.params.detach().cpu().numpy()

        boxes = []
        for i, sb in enumerate(self.sign_boxes_flat):
            boxes.append(SignBox(
                sign=sb.sign, score=sb.score,
                cx=float(p[i, 0]), cy=float(p[i, 1]),
                width=float(p[i, 2]), height=float(p[i, 3]),
                row_idx=sb.row_idx, col_idx=sb.col_idx,
            ))

        return SubTablet(
            img=self.sub_tablet_text.img,
            sign_boxes=boxes,
            name="optimized",
            scale_factor=self.sub_tablet_text.scale_factor,
            avg_width=self.sub_tablet_text.avg_width,
            avg_height=self.sub_tablet_text.avg_height,
            margin=self.sub_tablet_text.margin,
            origin_x=self.sub_tablet_text.origin_x,
            origin_y=self.sub_tablet_text.origin_y,
        )

    def get_param_changes(self) -> np.ndarray:
        """Return (current − initial) parameter array  (M, 4)."""
        return (self.params.detach().cpu().numpy()
                - self.initial_params.cpu().numpy())

    # ==================================================================
    #  Visualisation
    # ==================================================================

    def plot_loss_history(self, figsize: tuple = (20, 5)):
        """Plot loss curves: total, raw components, weighted components."""
        if not self.loss_components_history:
            print("No optimisation history available.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # -- Total loss --
        axes[0].plot(self.loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)

        # -- Raw component losses --
        keys = ['data', 'anchor', 'seq', 'height', 'rows']
        for k in keys:
            vals = [h[k] for h in self.loss_components_history]
            axes[1].plot(vals, label=f'L_{k}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Raw Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # -- Weighted components or sigma --
        sigma_vals = [h.get('sigma', self.sigma)
                      for h in self.loss_components_history]
        if len(set(sigma_vals)) > 1:
            ax2r = axes[2].twinx()
            for k in keys:
                lam = getattr(self, f'lambda_{k}')
                vals = [lam * h[k] for h in self.loss_components_history]
                axes[2].plot(vals, label=f'λ·L_{k}')
            ax2r.plot(sigma_vals, 'k--', alpha=0.5, label='σ')
            ax2r.set_ylabel('σ')
            ax2r.legend(loc='upper right')
            axes[2].set_title('Weighted Components + σ')
        else:
            for k in keys:
                lam = getattr(self, f'lambda_{k}')
                vals = [lam * h[k] for h in self.loss_components_history]
                axes[2].plot(vals, label=f'λ·L_{k}')
            axes[2].set_title('Weighted Loss Components')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Weighted Loss')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_topology(
        self,
        figsize: tuple = (16, 10),
        title: str = "Optimisation Point Set — Topology",
        show_labels: bool = True,
    ):
        """
        Standalone topology visualisation of the source point set S.

        * Rows are coloured distinctly.
        * Intra-row edges (adjacent signs) are drawn.
        * Sign class name is annotated above each box.
        * Row indices are labelled on the left.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        p = self.params.detach().cpu().numpy()

        cmap = plt.cm.tab10
        n_colors = max(self.num_rows, 1)

        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            color = cmap(ri / n_colors)

            # Intra-row edges
            for j in range(rl - 1):
                x1, y1 = p[off + j, 0], p[off + j, 1]
                x2, y2 = p[off + j + 1, 0], p[off + j + 1, 1]
                ax.plot([x1, x2], [y1, y2], '-', color=color,
                        linewidth=2, alpha=0.6)

            # Signs
            for j in range(rl):
                idx = off + j
                cx, cy, w, h = p[idx]
                sb = self.sign_boxes_flat[idx]

                rect = MplRectangle(
                    (cx - w / 2, cy - h / 2), w, h,
                    linewidth=1.5, edgecolor=color,
                    facecolor=color, alpha=0.15,
                )
                ax.add_patch(rect)
                ax.plot(cx, cy, 'o', color=color, markersize=4)

                if show_labels:
                    ax.text(cx, cy - h / 2 - 3, sb.sign_name[:8],
                            fontsize=6, ha='center', va='bottom',
                            color='black', fontweight='bold')

            # Row label
            if rl > 0:
                first_cx = p[off, 0]
                first_w = p[off, 2]
                first_cy = p[off, 1]
                ax.text(first_cx - first_w - 5, first_cy,
                        f'R{ri}', fontsize=10, ha='right', va='center',
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.25))

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_point_sets(
        self,
        img: np.ndarray = None,
        figsize: tuple = (16, 10),
        title: str = "Source S (blue) and Target X (red)",
    ):
        """
        Overlay both source (S) and target (X) point sets, optionally
        on the tablet image.

        * Source boxes: blue / cyan  (with intra-row connections)
        * Target boxes: red
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ---- Target X  (red) ----
        if self.N > 0:
            Xp = self.X_pos.cpu().numpy()
            Xs = self.X_size.cpu().numpy()
            for n in range(self.N):
                cx, cy = Xp[n]
                w, h = Xs[n]
                rect = MplRectangle(
                    (cx - w / 2, cy - h / 2), w, h,
                    linewidth=1, edgecolor='red',
                    facecolor='red', alpha=0.12,
                )
                ax.add_patch(rect)
            ax.scatter(Xp[:, 0], Xp[:, 1], c='red', s=20, zorder=5,
                       label=f'Target X ({self.N})')

        # ---- Source S  (blue / cyan) ----
        p = self.params.detach().cpu().numpy()
        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            for j in range(rl - 1):
                x1, y1 = p[off + j, 0], p[off + j, 1]
                x2, y2 = p[off + j + 1, 0], p[off + j + 1, 1]
                ax.plot([x1, x2], [y1, y2], '-', color='cyan',
                        linewidth=1, alpha=0.5)
            for j in range(rl):
                idx = off + j
                cx, cy, w, h = p[idx]
                rect = MplRectangle(
                    (cx - w / 2, cy - h / 2), w, h,
                    linewidth=1, edgecolor='blue',
                    facecolor='cyan', alpha=0.12,
                )
                ax.add_patch(rect)

        ax.scatter(p[:, 0], p[:, 1], c='blue', s=15, zorder=5,
                   marker='s', label=f'Source S ({self.M})')

        ax.legend(fontsize=10)
        ax.set_title(title)
        if img is None:
            ax.set_aspect('equal')
            ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
