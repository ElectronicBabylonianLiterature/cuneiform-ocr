"""
Point Set Registration (PSR) Optimizer for sign alignment.

GMM-based point set registration for aligning text-derived boxes to
detected boxes.  The source set S (text-derived signs, organized in rows)
is optimized to match the target set X (detected signs).

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

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from .sign import CLASSES_ABZ as DEFAULT_CLASSES_ABZ
from .box import Box, Boxes


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
        source_rows: List[List[Box]],
        target_detections: Boxes = None,
        classes_abz: List[str] = None,
        sigma: float = None,
        w_noise: float = 0.1,
        confusion_matrix: np.ndarray = None,
        lambda_data: float = 1.0,
        lambda_anchor: float = 0.1,
        lambda_seq: float = 0.1,
        lambda_height: float = 0.1,
        lambda_rows: float = 0.1,
        lambda_boundary: float = 0.0,
        boundary_steepness: float = 1000.0,
        rows_threshold_ratio_far: float = 1.0 / 3.0,
        rows_threshold_ratio_close: float = 1.0 / 2.0,
        rows_plateau_far: float = 1.0,
        rows_plateau_close: float = 1.0,
        contour_mask: np.ndarray = None,
        device: str = None,
    ):
        """
        Parameters
        ----------
        source_rows : list[list[Box]]
            Text-derived sign boxes grouped by row.
        target_detections : Boxes
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
        rows_threshold_ratio_far : float
            Fraction of ideal inter-row spacing that defines the
            quadratic → plateau transition in ``L_rows`` when rows
            are **too far apart** (positive deviation).
        rows_threshold_ratio_close : float
            Fraction of ideal inter-row spacing that defines the
            quadratic → plateau transition in ``L_rows`` when rows
            are **too close together** (negative deviation).
        rows_plateau_far : float
            Maximum (plateau) loss value for the **too far** side of
            ``L_rows``.
        rows_plateau_close : float
            Maximum (plateau) loss value for the **too close** side of
            ``L_rows``.
        lambda_boundary : float
            Weight for the contour boundary loss.  Set to 0 to disable
            (default).  Typical value: 5.0–20.0.
        boundary_steepness : float
            Steepness multiplier *k* for the boundary loss:
            ``L = k · (1 − IoR)²``.  Default 4.0.
        contour_mask : np.ndarray (H, W), optional
            Binary mask from ``divide_tablet_photo(return_masks=True)``.
            Values 255 = inside tablet, 0 = outside.  When provided
            together with ``lambda_boundary > 0``, an extra loss
            penalises the first sign of each row whose bounding box
            extends outside the contour (top / left / bottom edges).
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
        self.lambda_boundary = lambda_boundary
        self.boundary_steepness = boundary_steepness
        self.rows_threshold_ratio_far = rows_threshold_ratio_far
        self.rows_threshold_ratio_close = rows_threshold_ratio_close
        self.rows_plateau_far = rows_plateau_far
        self.rows_plateau_close = rows_plateau_close
        if source_rows is None:
            raise ValueError("PointSetRegistrationOptimizer requires source_rows")

        self.rows = [list(row) for row in source_rows if row]
        if not self.rows:
            raise ValueError("PointSetRegistrationOptimizer requires non-empty source_rows")
        self.source_tablet = self.rows[0][0].tablet
        self.source_boxes = Boxes((box for row in self.rows for box in row), tablet=self.source_tablet)

        # ---- Contour mask for boundary loss ------------------------------
        if contour_mask is not None and lambda_boundary > 0:
            self.contour_mask = contour_mask.astype(np.float32)
            if self.contour_mask.max() > 1:
                self.contour_mask = self.contour_mask / 255.0
        else:
            self.contour_mask = None

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # ---- Source point set S (text) -----------------------------------
        self.num_rows = len(self.rows)
        self.row_lengths = [len(r) for r in self.rows]

        self.source_boxes_flat: List[Box] = []
        self.row_indices: List[int] = []
        self.col_indices: List[int] = []
        self.class_ids_source: List[int] = []

        for row_idx, row in enumerate(self.rows):
            for col_idx, sb in enumerate(row):
                self.source_boxes_flat.append(sb)
                self.row_indices.append(row_idx)
                self.col_indices.append(col_idx)
                cid = (self.classes_abz.index(sb.sign.abz)
                       if sb.sign.abz in self.classes_abz else -1)
                self.class_ids_source.append(cid)

        self.M = len(self.source_boxes_flat)

        # ---- Target point set X (detections) -----------------------------
        self.target_detections = target_detections or Boxes(tablet=self.source_tablet)
        self.N = len(self.target_detections)

        target_pos = []
        target_sz = []
        self.class_ids_target: List[int] = []

        for det in self.target_detections:
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
                for sb in self.source_boxes_flat]
        self.params = torch.tensor(init, dtype=torch.float32,
                                   device=self.device, requires_grad=True)
        self.initial_params = self.params.clone().detach()

        # ---- Sigma -------------------------------------------------------
        avg_size = self.source_boxes.avg_size
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

        Asymmetric piecewise loss per pair:
            d = actual − ideal
            If d ≥ 0 (too far):  t = ideal · threshold_ratio_far,  plateau = plateau_far
            If d < 0 (too close): t = ideal · threshold_ratio_close, plateau = plateau_close
            |d| ≤ t :  L = plateau · (d/t)²
            |d| > t :  L = plateau
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
            # Asymmetric thresholds: different for too-far vs too-close
            threshold_far = (ideal * self.rows_threshold_ratio_far).clamp(min=1e-6)
            threshold_close = (ideal * self.rows_threshold_ratio_close).clamp(min=1e-6)
            threshold = torch.where(deviation >= 0, threshold_far, threshold_close)
            plateau = torch.where(
                deviation >= 0,
                torch.tensor(self.rows_plateau_far, device=self.device),
                torch.tensor(self.rows_plateau_close, device=self.device),
            )
            normalised = deviation / threshold

            # Piecewise: quadratic inside, constant plateau outside
            loss_pair = torch.where(
                normalised.abs() <= 1.0,
                plateau * normalised ** 2,
                plateau,
            )
            loss = loss + loss_pair
            count += 1

        return loss / max(1, count)

    # ------------------------------------------------------------------

    def compute_boundary_loss(self) -> torch.Tensor:
        """
        Penalise the first sign of each row when its bbox extends
        outside the tablet contour mask.

        For each row, take the first sign (leftmost in the row list).
        Compute the intersection area between its bbox and the contour
        mask.  The loss is based on  ``1 − IoR`` (Intersection-over-
        Region, where Region = bbox area), so a box fully inside the
        contour yields 0 loss.

        A steep exponential-like penalty is used so that even a small
        exceedance produces a strong gradient:

            L_boundary = mean_over_rows  [ (1 − IoR)^2 × k ]

        where k = 4 gives a steep curve.

        Returns ``0`` if no contour mask is set.
        """
        if self.contour_mask is None or self.num_rows == 0:
            return torch.tensor(0.0, device=self.device)

        mask_h, mask_w = self.contour_mask.shape[:2]
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        k = self.boundary_steepness

        for ri in range(self.num_rows):
            off = self.row_offsets[ri]
            rl = self.row_lengths[ri]
            if rl == 0:
                continue

            # First sign in the row (index 0, leftmost)
            cx = self.params[off, 0]
            cy = self.params[off, 1]
            w = self.params[off, 2]
            h = self.params[off, 3]

            # Bbox corners (float, differentiable)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clamp to image extent for the sampling grid
            x1_c = x1.clamp(0, mask_w - 1)
            y1_c = y1.clamp(0, mask_h - 1)
            x2_c = x2.clamp(0, mask_w - 1)
            y2_c = y2.clamp(0, mask_h - 1)

            # If the bbox is entirely outside the image, IoR = 0
            if (x1_c >= x2_c) or (y1_c >= y2_c):
                loss = loss + torch.tensor(k, device=self.device)
                count += 1
                continue

            # --- Compute IoR via grid sampling on the mask ----------------
            # Use a coarse grid of sample points inside the (clamped) bbox
            n_samples = 8  # per axis
            xs = torch.linspace(float(x1_c.detach()), float(x2_c.detach()),
                                n_samples, device=self.device)
            ys = torch.linspace(float(y1_c.detach()), float(y2_c.detach()),
                                n_samples, device=self.device)
            gx, gy = torch.meshgrid(xs, ys, indexing='xy')
            gx_int = gx.long().clamp(0, mask_w - 1)
            gy_int = gy.long().clamp(0, mask_h - 1)

            mask_t = torch.tensor(self.contour_mask, dtype=torch.float32,
                                  device=self.device)
            sampled = mask_t[gy_int, gx_int]  # (n, n) in [0,1]

            # Fraction of the clamped region that's inside the contour
            inside_ratio_clamped = sampled.mean()

            # Account for the part of the bbox that's outside the image
            # (which is definitely outside the contour)
            clamped_area = (x2_c - x1_c) * (y2_c - y1_c)
            bbox_area = (w * h).clamp(min=1.0)
            # What fraction of the full bbox is inside the image?
            area_ratio = clamped_area / bbox_area

            # Overall IoR: inside_clamped * (clamped_area / bbox_area)
            # Detach the sampling parts (non-differentiable lookup) and
            # keep the bbox geometry differentiable through area_ratio.
            ior = inside_ratio_clamped.detach() * area_ratio

            # Steep penalty:  k * (1 - IoR)^2
            violation = (1.0 - ior)
            loss = loss + k * violation ** 2
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
        L_boundary = self.compute_boundary_loss()

        L_total = (self.lambda_data * L_data
                   + self.lambda_anchor * L_anchor
                   + self.lambda_seq * L_seq
                   + self.lambda_height * L_height
                   + self.lambda_rows * L_rows
                   + self.lambda_boundary * L_boundary)

        return L_total, L_data, L_anchor, L_seq, L_height, L_rows, L_boundary

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
    ) -> Boxes:
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
        Boxes
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
                  f"rows={self.lambda_rows}, boundary={self.lambda_boundary}")
            if self.contour_mask is not None:
                print(f"  Contour mask: {self.contour_mask.shape[1]}×{self.contour_mask.shape[0]}")
            print("-" * 60)

        for it in range(num_iterations):
            # Optional sigma annealing
            if sigma_anneal:
                t = it / max(1, num_iterations - 1)
                self.sigma = sigma_init * (1.0 - t) + sigma_final * t

            opt.zero_grad()
            L_total, L_data, L_anchor, L_seq, L_height, L_rows, L_boundary = \
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
                'boundary': L_boundary.item(),
                'sigma': self.sigma,
            })

            if verbose and (it % log_every == 0 or it == num_iterations - 1):
                line = (f"Iter {it:4d}: total={L_total.item():.4f}  "
                        f"data={L_data.item():.4f}  "
                        f"anchor={L_anchor.item():.4f}  "
                        f"seq={L_seq.item():.4f}  "
                        f"height={L_height.item():.4f}  "
                        f"rows={L_rows.item():.4f}  "
                        f"boundary={L_boundary.item():.4f}")
                if sigma_anneal:
                    line += f"  σ={self.sigma:.1f}"
                print(line)

        return self.get_optimized_boxes()

    # ------------------------------------------------------------------
    #  Results
    # ------------------------------------------------------------------

    def get_optimized_boxes(self) -> Boxes:
        """Build the current optimised boxes."""
        p = self.params.detach().cpu().numpy()

        boxes = Boxes(tablet=self.source_boxes.tablet)
        for i, sb in enumerate(self.source_boxes_flat):
            boxes.append(Box.from_center(
                sign=sb.sign, score=sb.score,
                cx=float(p[i, 0]), cy=float(p[i, 1]),
                width=float(p[i, 2]), height=float(p[i, 3]),
                tablet=self.source_boxes.tablet,
            ))
        return boxes

    def get_param_changes(self) -> np.ndarray:
        """Return (current − initial) parameter array  (M, 4)."""
        return (self.params.detach().cpu().numpy()
                - self.initial_params.cpu().numpy())

    # ==================================================================
    #  Visualisation
    # ==================================================================

    def plot_loss_history(self, figsize: tuple = (16, 12)):
        """Plot loss curves in a 2x2 grid: total, raw, weighted, log-scale."""
        if not self.loss_components_history:
            print("No optimisation history available.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        keys = ['data', 'anchor', 'seq', 'height', 'rows']
        if self.lambda_boundary > 0:
            keys.append('boundary')

        # -- (0,0) Total loss --
        axes[0, 0].plot(self.loss_history)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # -- (0,1) Raw component losses --
        for k in keys:
            vals = [h[k] for h in self.loss_components_history]
            axes[0, 1].plot(vals, label=f'L_{k}')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Raw Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # -- (1,0) Weighted components (+ sigma if annealed) --
        sigma_vals = [h.get('sigma', self.sigma)
                      for h in self.loss_components_history]
        if len(set(sigma_vals)) > 1:
            ax_twin = axes[1, 0].twinx()
            for k in keys:
                lam = getattr(self, f'lambda_{k}')
                vals = [lam * h[k] for h in self.loss_components_history]
                axes[1, 0].plot(vals, label=f'λ·L_{k}')
            ax_twin.plot(sigma_vals, 'k--', alpha=0.5, label='σ')
            ax_twin.set_ylabel('σ')
            ax_twin.legend(loc='upper right')
            axes[1, 0].set_title('Weighted Components + σ')
        else:
            for k in keys:
                lam = getattr(self, f'lambda_{k}')
                vals = [lam * h[k] for h in self.loss_components_history]
                axes[1, 0].plot(vals, label=f'λ·L_{k}')
            axes[1, 0].set_title('Weighted Loss Components')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Weighted Loss')
        axes[1, 0].legend(loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)

        # -- (1,1) Log-scale raw components --
        for k in keys:
            vals = [h[k] for h in self.loss_components_history]
            axes[1, 1].plot(vals, label=f'L_{k}')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss (log)')
        axes[1, 1].set_title('Raw Loss Components (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_loss_curves(self, save_dir: str = "alignment_loss_functions",
                         show: bool = True):
        """
        Plot the characteristic shape of each loss function and save to files.

        Generates one image per loss function showing how the loss responds
        to its input variable, using the optimizer's current parameters.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []

        def finish(fig, filename: str) -> None:
            path = os.path.join(save_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files.append(path)
            if show:
                plt.show()
            else:
                plt.close(fig)

        # ---- 1. Rows loss (asymmetric piecewise) ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        r_far = self.rows_threshold_ratio_far
        r_close = self.rows_threshold_ratio_close
        p_far = self.rows_plateau_far
        p_close = self.rows_plateau_close
        # x-axis: deviation / ideal  (dimensionless ratio)
        d_ratio = np.linspace(-1.5, 1.5, 500)
        loss_vals = np.zeros_like(d_ratio)
        for i, dr in enumerate(d_ratio):
            if dr >= 0:
                t_r, plat = r_far, p_far
            else:
                t_r, plat = r_close, p_close
            normed = dr / t_r if t_r > 0 else 0
            loss_vals[i] = plat * min(normed ** 2, 1.0)
        ax.plot(d_ratio, loss_vals, 'b-', linewidth=2)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='ideal spacing')
        ax.axvline(r_far, color='r', linestyle=':', alpha=0.7,
                   label=f'threshold far = {r_far:.3f}')
        ax.axvline(-r_close, color='orange', linestyle=':', alpha=0.7,
                   label=f'threshold close = {r_close:.3f}')
        ax.axhline(p_far, color='r', linestyle='-', alpha=0.3,
                   label=f'plateau far = {p_far}')
        ax.axhline(p_close, color='orange', linestyle='-', alpha=0.3,
                   label=f'plateau close = {p_close}')
        ax.set_xlabel('(actual − ideal) / ideal')
        ax.set_ylabel('Loss')
        ax.set_title(f'$L_{{rows}}$: Asymmetric Piecewise Loss\n'
                     f'(r_far={r_far:.3f}, r_close={r_close:.3f}, '
                     f'p_far={p_far}, p_close={p_close})')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, max(p_far, p_close) * 1.3)
        finish(fig, 'loss_rows.png')

        # ---- 2. Anchor loss (quadratic deviation from baseline) ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        dev = np.linspace(-100, 100, 500)
        loss_a = dev ** 2
        ax.plot(dev, loss_a, 'b-', linewidth=2)
        ax.set_xlabel('y − y_baseline  (pixels)')
        ax.set_ylabel('Loss (per sign)')
        ax.set_title('$L_{anchor}$: Squared Deviation from Row Baseline')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        finish(fig, 'loss_anchor.png')

        # ---- 3. Seq loss (gap deviation) ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        gap_dev = np.linspace(-100, 100, 500)
        loss_s = gap_dev ** 2
        ax.plot(gap_dev, loss_s, 'b-', linewidth=2)
        ax.set_xlabel('actual_gap − expected_gap  (pixels)')
        ax.set_ylabel('Loss (per pair)')
        ax.set_title('$L_{seq}$: Squared Gap Deviation\n'
                     '(expected_gap = $(w_j + w_{j+1})/2$)')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5,
                   label='ideal gap')
        ax.legend()
        finish(fig, 'loss_seq.png')

        # ---- 4. Height loss (variance) ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # Show how variance grows as one sign deviates from mean
        # In a row of n signs, if one deviates by delta from the
        # uniform height, Var = delta^2 * (n-1)/n^2
        # For simplicity show Var for n=5
        n_demo = 5
        delta = np.linspace(-80, 80, 500)
        var_vals = delta ** 2 * (n_demo - 1) / (n_demo ** 2)
        ax.plot(delta, var_vals, 'b-', linewidth=2,
                label=f'row with {n_demo} signs')
        for n_ex in [3, 10]:
            v = delta ** 2 * (n_ex - 1) / (n_ex ** 2)
            ax.plot(delta, v, '--', linewidth=1, alpha=0.6,
                    label=f'row with {n_ex} signs')
        ax.set_xlabel('$h_j - \\bar{h}$  (pixel deviation of one sign)')
        ax.set_ylabel('Variance contribution')
        ax.set_title('$L_{height}$: Height Variance within a Row\n'
                     '(one sign deviating, others uniform)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        finish(fig, 'loss_height.png')

        # ---- 5. Data loss (GMM likelihood for one source) ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sigma = self.sigma
        w = self.w_noise
        # Show -log p(x) as a function of distance from the nearest
        # source centre, for a single source-target pair
        dist = np.linspace(0, 4 * sigma, 500)
        gauss = (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(
            -dist ** 2 / (2 * sigma ** 2))
        # p(x) = w/N + (1-w)/M * gauss  (simplified for M=N=1)
        p_x = w + (1 - w) * gauss
        neg_log_p = -np.log(p_x)
        ax.plot(dist, neg_log_p, 'b-', linewidth=2)
        ax.axhline(-np.log(w), color='r', linestyle=':', alpha=0.7,
                   label=f'floor: −log(w) = {-np.log(w):.2f}')
        ax.axvline(sigma, color='orange', linestyle='--', alpha=0.6,
                   label=f'σ = {sigma:.1f}')
        ax.axvline(2 * sigma, color='orange', linestyle=':', alpha=0.4,
                   label=f'2σ = {2*sigma:.1f}')
        ax.set_xlabel('‖x − s‖  (distance, pixels)')
        ax.set_ylabel('−log p(x)')
        ax.set_title(f'$E_{{data}}$: Neg-Log-Likelihood per Target Point\n'
                     f'(σ={sigma:.1f}, w={w})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        finish(fig, 'loss_data.png')

        # ---- 6. Boundary loss (steep IoR penalty) ----
        if self.lambda_boundary > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ior_vals = np.linspace(0, 1, 500)
            k = self.boundary_steepness
            boundary_loss = k * (1.0 - ior_vals) ** 2
            ax.plot(ior_vals, boundary_loss, 'b-', linewidth=2)
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5,
                       label='fully inside (IoR=1)')
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            # Mark some reference points
            for ref_ior in [0.5, 0.8, 0.9]:
                ref_loss = k * (1.0 - ref_ior) ** 2
                ax.plot(ref_ior, ref_loss, 'ro', markersize=5)
                ax.annotate(f'IoR={ref_ior}: L={ref_loss:.2f}',
                           xy=(ref_ior, ref_loss),
                           xytext=(ref_ior - 0.15, ref_loss + 0.2),
                           fontsize=8, alpha=0.8)
            ax.set_xlabel('IoR (Intersection over Region)')
            ax.set_ylabel('Loss (per row-first sign)')
            ax.set_title(f'$L_{{boundary}}$: Contour Boundary Penalty\n'
                         f'$k \\cdot (1 - IoR)^2$, k={k}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.1, k + 0.5)
            finish(fig, 'loss_boundary.png')

        print(f"Loss curve plots saved to {save_dir}/:")
        for f in saved_files:
            print(f"  {f}")
        return saved_files
