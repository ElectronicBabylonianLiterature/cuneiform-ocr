import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from typing import List, Optional, Callable

from .sign import SignResolver, CLASSES_ABZ
from .bounding_box import BoundingBox, Detection
from .tablet import SignBox


class BboxVisualizer:
    """Visualize bounding boxes on images."""
    
    def __init__(self, color=(0, 255, 0)):
        """
        Args:
            color: Default box color in RGB format
        """
        self.default_color = color
        self.color_func: Optional[Callable[[BoundingBox], tuple]] = None
        self.result = None  # Store last result for display
    
    def draw_boxes(self, img: np.ndarray, boxes: Detection, show_labels: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            img: Input image in BGR format
            boxes: List of BoundingBox objects
            show_labels: Whether to show sign name labels
            
        Returns:
            Image with boxes drawn
        """
        img_vis = img.copy()
        
        # Draw rectangles
        for box in boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            
            # Get color
            color = self.color_func(box) if self.color_func else self.default_color
            color_bgr = tuple(reversed(color))  # RGB to BGR
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color_bgr, 2)
        
        if not show_labels:
            self.result = img_vis
            return img_vis
        
        # Convert to PIL for Unicode text
        img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Draw labels
        for box in boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            label = box.sign.name[:10]
            
            # Get box color
            # color = self.color_func(box) if self.color_func else self.default_color
            color = (0, 0, 0) # still use black
            
            # Calculate label height as 1/6 of box height
            box_height = y2 - y1
            label_height = max(int(box_height / 6), 12)  # Minimum 12 pixels
            
            # Calculate font size to fit label height (approximate: font_size ≈ label_height * 0.75)
            font_size = max(int(label_height * 0.75), 10)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            
            # Position label inside box at the top
            label_x1 = x1 + 2
            label_y1 = y1 + 2
            label_x2 = min(x1 + text_w + 8, x2 - 2)  # Ensure it stays within box
            label_y2 = y1 + label_height
            
            # Draw label background with box color
            draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
            
            # Draw white text
            text_y = label_y1 + (label_height - text_h) // 2  # Center vertically
            draw.text((label_x1 + 4, text_y), label, font=font, fill=(255, 255, 255))
        
        img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.result = img_result  # Store for display_result
        return img_result
    
    def display_result(self, vis_opt: str = "show", path: str = None):
        """
        Display or save the visualization result.
        
        Args:
            vis_opt: "show" for cv2.imshow, "draw" for matplotlib, "save" for file
            path: File path when vis_opt="save"
        """
        if self.result is None:
            print("No result to display. Call draw_boxes first.")
            return
        
        if vis_opt == "show":
            cv2.imshow('Visualization', self.result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif vis_opt == "draw":
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        elif vis_opt == "save":
            if path is None:
                path = 'bbox_visualization.jpg'
                print(f"No path provided, using default: {path}")
            
            # Create directory if needed
            import os
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            cv2.imwrite(path, self.result)
            abs_path = os.path.abspath(path)
            print(f"✓ Saved to: {abs_path}")
    
    def save(self, path: str):
        """Save result to file."""
        self.display_result(vis_opt="save", path=path)
    
    def show_draw(self):
        """Display result using matplotlib."""
        self.display_result(vis_opt="draw")
    
    def draw_rows(
        self, 
        img: np.ndarray, 
        boxes: List,
        show_labels: bool = True,
        show_row_numbers: bool = False,
        row_mapping: dict = None,
        line_thickness: int = 2,
        marker_size: int = 5
    ) -> np.ndarray:
        """
        Draw bounding boxes with row connections.
        
        First uses draw_boxes to draw the bounding boxes,
        then draws lines connecting center points of signs in the same row.
        Each row gets a unique color.
        
        Args:
            img: Input image in BGR format (None for text-only visualization)
            boxes: List of box objects with row_idx attribute (SignBox, BoundingBox with row_idx, etc.)
            show_labels: Whether to show sign name labels
            show_row_numbers: Whether to annotate row numbers on the left margin
            row_mapping: Optional dict mapping row_idx to display number (for matched rows)
            line_thickness: Thickness of row connection lines
            marker_size: Size of center point markers
            
        Returns:
            Image with boxes and row connections drawn
        """
        # Define distinct colors for rows (HSV-based for better distinction)
        def get_row_color(row_idx: int) -> tuple:
            """Generate distinct color for row using HSV (returns RGB)."""
            if row_idx == -1:
                return (128, 128, 128)  # Gray for noise
            hue = int((row_idx * 137.5) % 180)  # Golden angle for better distribution
            sat = 255
            val = 255
            hsv_color = np.uint8([[[hue, sat, val]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            rgb_color = tuple(reversed(bgr_color))  # Convert BGR to RGB
            return tuple(map(int, rgb_color))
        
        def get_effective_row_idx(row_idx: int) -> int:
            """Get effective row index for coloring (considering row_mapping)."""
            if row_mapping is None:
                return row_idx
            # If row_mapping exists but this row is not in it, mark as unmapped (-2)
            if row_idx not in row_mapping:
                return -2  # Special code for unmapped rows
            # Use the mapped detection row index
            return row_mapping[row_idx]
        
        def get_row_color_with_mapping(row_idx: int) -> tuple:
            """Get color for row, considering row_mapping if provided."""
            effective_idx = get_effective_row_idx(row_idx)
            if effective_idx == -2:
                return (128, 128, 128)  # Dark gray for unmapped rows
            return get_row_color(effective_idx)
        
        # Handle text-only visualization (no background image)
        if img is None:
            if not boxes:
                return np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Create white canvas based on box extents
            # Need to handle negative coordinates from centroid alignment
            max_x = max(b.cx + b.width / 2 if hasattr(b, 'cx') else b.x2 for b in boxes)
            max_y = max(b.cy + b.height / 2 if hasattr(b, 'cy') else b.y2 for b in boxes)
            min_x = min(b.cx - b.width / 2 if hasattr(b, 'cx') else b.x1 for b in boxes)
            min_y = min(b.cy - b.height / 2 if hasattr(b, 'cy') else b.y1 for b in boxes)
            
            margin = 100
            
            # Calculate offset needed to shift negative coordinates into visible area
            offset_x = max(0, -min_x) + margin
            offset_y = max(0, -min_y) + margin
            
            canvas_width = int(max_x - min_x + 2 * margin)
            canvas_height = int(max_y - min_y + 2 * margin)
            
            img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Create shifted copies of boxes for visualization (don't modify originals)
            boxes_shifted = []
            for box in boxes:
                if isinstance(box, SignBox):
                    # Create new SignBox with adjusted center coordinates
                    # x1, y1, x2, y2 will be automatically calculated from cx, cy, width, height
                    box_shifted = SignBox(
                        sign=box.sign,
                        score=box.score,
                        cx=box.cx + offset_x,
                        cy=box.cy + offset_y,
                        width=box.width,
                        height=box.height,
                        row_idx=box.row_idx,
                        col_idx=box.col_idx
                    )
                else:
                    # For other box types, create a copy and adjust if possible
                    import copy
                    box_shifted = copy.copy(box)
                    if hasattr(box_shifted, 'cx'):
                        box_shifted.cx = box.cx + offset_x
                        box_shifted.cy = box.cy + offset_y
                boxes_shifted.append(box_shifted)
            
            # Use shifted boxes for visualization
            boxes = boxes_shifted
        
        # Set color function to color boxes by row
        self.color_func = lambda box: get_row_color_with_mapping(getattr(box, 'row_idx', -1))
        
        # Convert boxes to Detection format (BoundingBox list) if needed
        detection_boxes = []
        for box in boxes:
            if hasattr(box, 'to_bounding_box'):
                # SignBox
                detection_boxes.append(box.to_bounding_box())
            else:
                # Already BoundingBox
                detection_boxes.append(box)
            # Copy row_idx to the detection box
            if hasattr(box, 'row_idx'):
                detection_boxes[-1].row_idx = box.row_idx
        
        # Use draw_boxes to draw bounding boxes with labels
        img_vis = self.draw_boxes(img, detection_boxes, show_labels=show_labels)
        
        # Reset color function
        self.color_func = None
        
        # Group boxes by row
        rows = {}
        for box in boxes:
            row_idx = getattr(box, 'row_idx', -1)
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(box)
        
        # Draw row connections on top of the boxes
        for row_idx in sorted(rows.keys()):
            if row_idx == -1:
                continue  # Skip noise
            
            row_boxes = rows[row_idx]
            if len(row_boxes) < 2:
                continue  # Need at least 2 boxes to draw line
            
            # Sort boxes by x-coordinate (left to right)
            def get_cx(b):
                if hasattr(b, 'cx'):
                    return b.cx
                elif hasattr(b, 'center'):
                    return b.center[0]
                else:
                    return (b.x1 + b.x2) / 2
            
            sorted_boxes = sorted(row_boxes, key=get_cx)
            
            # Get row color in BGR for OpenCV
            row_color_rgb = get_row_color_with_mapping(row_idx)
            row_color_bgr = tuple(reversed(row_color_rgb))
            
            # Draw lines connecting centers
            for i in range(len(sorted_boxes) - 1):
                box1 = sorted_boxes[i]
                box2 = sorted_boxes[i + 1]
                
                # Get centers
                def get_center(b):
                    if hasattr(b, 'cx') and hasattr(b, 'cy'):
                        return (int(b.cx), int(b.cy))
                    elif hasattr(b, 'center'):
                        return (int(b.center[0]), int(b.center[1]))
                    else:
                        return (int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2))
                
                cx1, cy1 = get_center(box1)
                cx2, cy2 = get_center(box2)
                
                # Draw line
                cv2.line(img_vis, (cx1, cy1), (cx2, cy2), row_color_bgr, line_thickness)
            
            # Draw center point markers for all boxes in row
            for box in sorted_boxes:
                if hasattr(box, 'cx') and hasattr(box, 'cy'):
                    cx, cy = int(box.cx), int(box.cy)
                elif hasattr(box, 'center'):
                    cx, cy = int(box.center[0]), int(box.center[1])
                else:
                    cx = int((box.x1 + box.x2) / 2)
                    cy = int((box.y1 + box.y2) / 2)
                
                cv2.circle(img_vis, (cx, cy), marker_size, row_color_bgr, -1)
                cv2.circle(img_vis, (cx, cy), marker_size + 1, (255, 255, 255), 1)  # White border
        
        # Add row number annotations if requested
        if show_row_numbers:
            # Convert to PIL for Unicode support (arrow character)
            img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                except:
                    font = ImageFont.load_default()
            
            for row_idx in sorted(rows.keys()):
                if row_idx == -1:
                    continue
                
                row_boxes = rows[row_idx]
                if not row_boxes:
                    continue
                
                # Compute average y position for this row
                avg_y = np.mean([
                    b.cy if hasattr(b, 'cy') else (b.y1 + b.y2) / 2 
                    for b in row_boxes
                ])
                
                # Get row color
                row_color_rgb = get_row_color_with_mapping(row_idx)
                
                # Determine display text
                if row_mapping is not None and row_idx in row_mapping:
                    mapped_row = row_mapping[row_idx]
                    # Show both: "R{text_row}→D{det_row}"
                    label_text = f"R{row_idx}→D{mapped_row}"
                else:
                    label_text = f"R{row_idx}"
                
                # Draw text with PIL (supports Unicode arrow)
                label_pos = (10, int(avg_y) - 12)  # Adjust y to center text vertically
                draw.text(label_pos, label_text, font=font, fill=row_color_rgb)
            
            # Convert back to OpenCV format
            img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        self.result = img_vis
        return img_vis


class TextVisualizer:
    """Visualize text lines."""
    
    @staticmethod
    def save_text(text_lines: List[List[str]], path: str, fragment_id: str = None):
        """
        Save text lines to file.
        
        Args:
            text_lines: List of lines, each containing sign names
            path: Output file path
            fragment_id: Optional fragment identifier
        """
        with open(path, 'w', encoding='utf-8') as f:
            if fragment_id:
                f.write(f"Fragment: {fragment_id}\n")
                f.write("=" * 50 + "\n")
            for i, line in enumerate(text_lines, 1):
                f.write(f"Line {i}: {' '.join(line)}\n")


class HeatmapVisualizer:
    """Visualize heatmaps with optional image/detection overlay."""
    
    def __init__(self, bbox_color=(255, 0, 0)):
        """
        Args:
            bbox_color: Color for bounding boxes in RGB format
        """
        self.bbox_color = bbox_color
        self.fig = None  # Store last figure for display
    
    def draw_channels(
        self, 
        img: Optional[np.ndarray], 
        heatmap: np.ndarray, 
        channels: tuple = (0, 1, 2),
        detection: Optional[Detection] = None,
        text_lines: Optional[List[List[str]]] = None
    ) -> plt.Figure:
        """
        Draw specific heatmap channels with image/detection overlay.
        
        Args:
            img: Background image (BGR), can be None for text-only
            heatmap: Heatmap array (H, W, num_classes)
            channels: Channel indices to visualize
            detection: Optional bounding boxes to show
            text_lines: Optional text lines to display
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # First subplot: image/detection/text
        if text_lines is not None:
            background = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8) * 255
            axes[0, 0].imshow(background)
            axes[0, 0].set_title('Text Lines')
            axes[0, 0].axis('off')
            
            y_start = 0.95
            y_step = 0.85 / max(len(text_lines), 1)
            for i, line in enumerate(text_lines):
                text = ' '.join(line)
                axes[0, 0].text(0.05, y_start - i * y_step, text,
                              fontsize=10, color='black', family='monospace',
                              transform=axes[0, 0].transAxes, verticalalignment='top')
            img = background
        elif detection is not None and img is not None:
            viz = BboxVisualizer(color=self.bbox_color)
            img_with_boxes = viz.draw_boxes(img, detection)
            axes[0, 0].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Detection')
            axes[0, 0].axis('off')
        elif img is not None:
            axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Image')
            axes[0, 0].axis('off')
        else:
            axes[0, 0].axis('off')
        
        # Heatmap channels
        for i in range(min(len(channels), 3)):
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            ch_idx = channels[i]
            heatmap_ch = heatmap[:, :, ch_idx]
            
            # Overlay on resized image
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
                axes[row, col].imshow(img_resized, alpha=0.5)
            
            vmax = 1.0
            ch_max = heatmap_ch.max()
            im = axes[row, col].imshow(heatmap_ch, cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
            
            # Title
            if ch_idx < len(CLASSES_ABZ):
                sign = SignResolver.from_idx(ch_idx)
                title = f'Ch {ch_idx}: {sign.abz} → {sign.name}\nmax={ch_max:.3f}'
            else:
                title = f'Ch {ch_idx}: max={ch_max:.3f}'
            
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        plt.tight_layout()
        self.fig = fig  # Store for display_result
        return fig
    
    def draw_pca(
        self,
        img: Optional[np.ndarray],
        heatmap: np.ndarray,
        n_components: int = 3,
        detection: Optional[Detection] = None,
        text_lines: Optional[List[List[str]]] = None,
        alpha: float = 0.6
    ) -> plt.Figure:
        """
        Visualize heatmap using PCA (first 3 components as RGB).
        
        Args:
            img: Background image (BGR), can be None
            heatmap: Heatmap array (H, W, num_classes)
            n_components: Number of PCA components
            detection: Optional bounding boxes
            text_lines: Optional text lines
            alpha: Overlay transparency
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare background
        if text_lines is not None:
            background = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8) * 255
            img = background
        
        # PCA
        H, W, C = heatmap.shape
        heatmap_flat = heatmap.reshape(-1, C)
        pca = PCA(n_components=min(n_components, C))
        pca_result = pca.fit_transform(heatmap_flat)
        pca_spatial = pca_result.reshape(H, W, -1)
        
        # Create false RGB
        false_rgb = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(min(3, pca_spatial.shape[2])):
            false_rgb[:, :, i] = pca_spatial[:, :, i]
        
        # Normalize
        false_rgb_norm = np.zeros_like(false_rgb)
        for i in range(3):
            ch_min = false_rgb[:, :, i].min()
            ch_max = false_rgb[:, :, i].max()
            if ch_max > ch_min:
                false_rgb_norm[:, :, i] = (false_rgb[:, :, i] - ch_min) / (ch_max - ch_min)
        
        explained_var = pca.explained_variance_ratio_[:3]
        
        # Subplot 1: Source (image/detection/text)
        if text_lines is not None:
            axes[0].imshow(background)
            axes[0].set_title('Text Lines')
            axes[0].axis('off')
            y_start = 0.95
            y_step = 0.85 / max(len(text_lines), 1)
            for i, line in enumerate(text_lines):
                text = ' '.join(line)
                axes[0].text(0.05, y_start - i * y_step, text,
                           fontsize=10, color='black', family='monospace',
                           transform=axes[0].transAxes, verticalalignment='top')
        elif detection is not None and img is not None:
            viz = BboxVisualizer(color=self.bbox_color)
            img_with_boxes = viz.draw_boxes(img, detection)
            axes[0].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Detection')
            axes[0].axis('off')
        elif img is not None:
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Image')
            axes[0].axis('off')
        else:
            axes[0].axis('off')
        
        # Subplot 2: PCA heatmap
        axes[1].imshow(false_rgb_norm)
        title = (f'PCA False RGB\n'
                f'R: PC1 ({explained_var[0]:.1%}) '
                f'G: PC2 ({explained_var[1]:.1%}) '
                f'B: PC3 ({explained_var[2]:.1%})')
        axes[1].set_title(title)
        axes[1].axis('off')
        
        # Subplot 3: Overlay
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
            img_norm = img_resized.astype(np.float32) / 255.0
            blended = img_norm * (1 - alpha) + false_rgb_norm * alpha
            blended = np.clip(blended, 0, 1)
            axes[2].imshow(blended)
            axes[2].set_title(f'Overlay (alpha={alpha})')
        else:
            axes[2].imshow(false_rgb_norm)
            axes[2].set_title('PCA Heatmap')
        axes[2].axis('off')
        
        plt.tight_layout()
        self.fig = fig  # Store for display_result
        return fig
    
    def display_result(self, vis_opt: str = "show", path: str = None, dpi: int = 150):
        """
        Display or save the heatmap visualization.
        
        Args:
            vis_opt: "show" or "draw" for display, "save" for file
            path: File path when vis_opt="save"
            dpi: DPI for saved figure
        """
        if self.fig is None:
            print("No figure to display. Call draw_channels or draw_pca first.")
            return
        
        if vis_opt in ("show", "draw"):
            plt.show()
        elif vis_opt == "save":
            if path is None:
                path = 'heatmap_visualization.jpg'
                print(f"No path provided, using default: {path}")
            
            # Create directory if needed
            import os
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            self.fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(self.fig)
            abs_path = os.path.abspath(path)
            print(f"✓ Saved to: {abs_path}")
    
    @staticmethod
    def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
        """Save a matplotlib figure to file."""
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def show_figure(fig: plt.Figure):
        """Display a matplotlib figure."""
        plt.show()
