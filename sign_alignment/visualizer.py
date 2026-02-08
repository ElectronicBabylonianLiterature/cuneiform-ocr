import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from typing import List, Optional, Callable

from .sign import SignResolver, CLASSES_ABZ
from .bounding_box import BoundingBox, Detection


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
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw labels
        for box in boxes:
            x1, y1 = int(box.x1), int(box.y1)
            label = box.sign.name[:10]
            
            # Get text size
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            
            # Draw background and text
            draw.rectangle([x1, y1-text_h-10, x1+text_w+4, y1-2], fill=(0, 0, 0))
            draw.text((x1+2, y1-text_h-8), label, font=font, fill=(255, 255, 255))
        
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
