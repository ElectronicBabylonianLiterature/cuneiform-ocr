import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Callable

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
        row_label_prefix: str = "R",
        mapped_label_prefix: str = "D",
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
            row_mapping: Optional dict mapping row_idx to mapped row idx (for matched rows)
            row_label_prefix: Prefix for row labels ("R" for text rows, "D" for detection rows)
            mapped_label_prefix: Prefix for the mapped row in labels ("D" for detection, "R" for text)
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
                
                # Determine display text (1-indexed for display)
                display_idx = row_idx + 1
                if row_mapping is not None and row_idx in row_mapping:
                    mapped_row = row_mapping[row_idx]
                    mapped_display = mapped_row + 1
                    label_text = f"{row_label_prefix}{display_idx}→{mapped_label_prefix}{mapped_display}"
                else:
                    label_text = f"{row_label_prefix}{display_idx}"
                
                # Draw text with PIL (supports Unicode arrow)
                label_pos = (10, int(avg_y) - 12)  # Adjust y to center text vertically
                draw.text(label_pos, label_text, font=font, fill=row_color_rgb)
            
            # Convert back to OpenCV format
            img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        self.result = img_vis
        return img_vis

    def draw_text_mapping(
        self,
        img: Optional[np.ndarray],
        sign_boxes: List,
        row_mapping: dict,
        sign_match_info: dict,
        mapped_label_prefix: str = "D",
        line_thickness: int = 2,
        marker_size: int = 5
    ) -> np.ndarray:
        """
        Draw text rows with per-sign match-quality coloring and supplementary labels.

        Shows all text rows with coloring reflecting how each sign matched to a
        detection: primary color for same-label match, desaturated for diff-label,
        gray for unmatched signs or unmatched rows.

        Args:
            img: Background image in BGR (None → auto white canvas)
            sign_boxes: List of SignBox from text subtablet
            row_mapping: text_to_det dict {text_row_idx: det_row_idx}
            sign_match_info: dict {(row_idx, col_idx): {"status": "same"|"diff"|"unmatched",
                             "det_sign_name": str|None}}
            mapped_label_prefix: Prefix for mapped rows ("D")
            line_thickness: Row connection line thickness
            marker_size: Center marker radius

        Returns:
            Image with text mapping visualization drawn
        """
        boxes = list(sign_boxes)

        # Handle text-only canvas (no image)
        if img is None:
            if not boxes:
                self.result = np.ones((100, 100, 3), dtype=np.uint8) * 255
                return self.result
            max_x = max(b.cx + b.width / 2 for b in boxes)
            max_y = max(b.cy + b.height / 2 for b in boxes)
            min_x = min(b.cx - b.width / 2 for b in boxes)
            min_y = min(b.cy - b.height / 2 for b in boxes)
            margin = 100
            offset_x = max(0, -min_x) + margin
            offset_y = max(0, -min_y) + margin
            canvas_w = int(max_x - min_x + 2 * margin)
            canvas_h = int(max_y - min_y + 2 * margin)
            img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
            new_boxes = []
            for box in boxes:
                new_boxes.append(SignBox(
                    sign=box.sign, score=box.score,
                    cx=box.cx + offset_x, cy=box.cy + offset_y,
                    width=box.width, height=box.height,
                    row_idx=box.row_idx, col_idx=box.col_idx
                ))
            boxes = new_boxes

        img_vis = img.copy()

        # Group boxes by row
        rows = {}
        for box in boxes:
            rows.setdefault(box.row_idx, []).append(box)

        # --- Draw rectangles colored by match status ---
        for box in boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            key = (box.row_idx, box.col_idx)
            info = sign_match_info.get(key, {"status": "unmatched", "det_sign_name": None})
            color_rgb = self._get_sign_color(box.row_idx, info["status"], row_mapping)
            color_bgr = tuple(reversed(color_rgb))
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color_bgr, 2)

        # --- Draw center lines and markers per row ---
        for row_idx in sorted(rows.keys()):
            if row_idx == -1:
                continue
            row_boxes = sorted(rows[row_idx], key=lambda b: b.cx)
            if row_idx in row_mapping:
                line_color_rgb = _get_row_color(row_idx)
            else:
                line_color_rgb = (128, 128, 128)
            line_color_bgr = tuple(reversed(line_color_rgb))
            for i in range(len(row_boxes) - 1):
                c1 = (int(row_boxes[i].cx), int(row_boxes[i].cy))
                c2 = (int(row_boxes[i + 1].cx), int(row_boxes[i + 1].cy))
                cv2.line(img_vis, c1, c2, line_color_bgr, line_thickness)
            for box in row_boxes:
                c = (int(box.cx), int(box.cy))
                cv2.circle(img_vis, c, marker_size, line_color_bgr, -1)
                cv2.circle(img_vis, c, marker_size + 1, (255, 255, 255), 1)

        # --- Draw labels with PIL (Unicode support) ---
        img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_font(24)
        label_font = _get_font(10)

        for box in boxes:
            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)
            key = (box.row_idx, box.col_idx)
            info = sign_match_info.get(key, {"status": "unmatched", "det_sign_name": None})
            status = info["status"]

            box_height = y2 - y1
            lbl_h = max(int(box_height / 6), 12)
            fsize = max(int(lbl_h * 0.75), 10)
            lbl_font = _get_font(fsize)

            label = box.sign_name[:10] if hasattr(box, 'sign_name') else str(box.sign.name[:10])

            # Primary label (text sign name)
            bbox_text = draw.textbbox((0, 0), label, font=lbl_font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            lx1, ly1 = x1 + 2, y1 + 2
            lx2, ly2 = min(x1 + tw + 8, x2 - 2), y1 + lbl_h
            draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0))
            text_y = ly1 + (lbl_h - th) // 2
            draw.text((lx1 + 4, text_y), label, font=lbl_font, fill=(255, 255, 255))

            # Supplementary label for diff-label matches
            if status == "diff" and info.get("det_sign_name"):
                det_label = info["det_sign_name"][:10]
                bbox_det = draw.textbbox((0, 0), det_label, font=lbl_font)
                dtw = bbox_det[2] - bbox_det[0]
                dth = bbox_det[3] - bbox_det[1]
                dlx1, dly1 = x1 + 2, ly2 + 1
                dlx2, dly2 = min(x1 + dtw + 8, x2 - 2), ly2 + 1 + lbl_h
                draw.rectangle([dlx1, dly1, dlx2, dly2], fill=(80, 80, 80))
                det_text_y = dly1 + (lbl_h - dth) // 2
                draw.text((dlx1 + 4, det_text_y), det_label, font=lbl_font, fill=(255, 255, 255))

        # --- Row annotations on left margin ---
        for row_idx in sorted(rows.keys()):
            if row_idx == -1:
                continue
            row_boxes = rows[row_idx]
            avg_y = np.mean([b.cy for b in row_boxes])
            display_idx = row_idx + 1
            if row_idx in row_mapping:
                mapped_display = row_mapping[row_idx] + 1
                lbl = f"R{display_idx}→{mapped_label_prefix}{mapped_display}"
                color_rgb = _get_row_color(row_idx)
            else:
                lbl = f"R{display_idx}"
                color_rgb = (128, 128, 128)
            draw.text((10, int(avg_y) - 12), lbl, font=font, fill=color_rgb)

        img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.result = img_vis
        return img_vis

    def draw_alignment_diagnostic(
        self,
        img: np.ndarray,
        detection_sign_boxes: List,
        aligned_text_boxes: List,
        det_sign_match_info: dict,
        text_sign_match_info: dict,
        det_to_text: dict,
        line_thickness: int = 2,
        marker_size: int = 5
    ) -> np.ndarray:
        """
        Draw alignment diagnostic on the detection image.

        Shows detection boxes colored by match quality, overlays coarse-aligned
        text boxes (unmatched ones as dashed rectangles), draws detection center
        lines (solid) and text center lines (dashed), with row annotations
        D#→R#.

        Args:
            img: Background image in BGR format
            detection_sign_boxes: List of SignBox from detection subtablet
            aligned_text_boxes: List of SignBox from coarse-aligned subtablet (sub_tablet_optim)
            det_sign_match_info: {(det_row_idx, det_col_idx): {"status": "same"|"diff"|"unmatched",
                                  "text_sign_name": str|None}}
            text_sign_match_info: {(text_row_idx, text_col_idx): {"status": "same"|"diff"|"unmatched",
                                   "det_sign_name": str|None}}
            det_to_text: dict {det_row_idx: text_row_idx}
            line_thickness: Line thickness
            marker_size: Marker radius

        Returns:
            Image with alignment diagnostic drawn
        """
        img_vis = img.copy()

        # Helper: map det row index through det_to_text for consistent coloring with draw_rows
        def _color_idx(det_row_idx: int) -> int:
            return det_to_text.get(det_row_idx, det_row_idx)

        # Group by row
        det_rows = {}
        for box in detection_sign_boxes:
            det_rows.setdefault(box.row_idx, []).append(box)
        text_rows = {}
        for box in aligned_text_boxes:
            text_rows.setdefault(box.row_idx, []).append(box)

        # --- Draw detection boxes colored by match status ---
        for box in detection_sign_boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            key = (box.row_idx, box.col_idx)
            info = det_sign_match_info.get(key, {"status": "unmatched", "text_sign_name": None})
            color_rgb = self._get_det_sign_color(_color_idx(box.row_idx), info["status"])
            color_bgr = tuple(reversed(color_rgb))
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color_bgr, 2)

        # --- Draw aligned text boxes (non-same) as dashed rectangles ---
        for box in aligned_text_boxes:
            key = (box.row_idx, box.col_idx)
            info = text_sign_match_info.get(key, {"status": "unmatched", "det_sign_name": None})
            if info["status"] == "diff":
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                color_rgb = _desaturate_color(_get_row_color(box.row_idx), 0.4)
                color_bgr = tuple(reversed(color_rgb))
                _draw_dashed_rect(img_vis, (x1, y1), (x2, y2), color_bgr, 2, 8)
            elif info["status"] == "unmatched":
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                _draw_dashed_rect(img_vis, (x1, y1), (x2, y2), (180, 180, 180), 2, 8)

        # --- Detection center lines (solid) ---
        for row_idx in sorted(det_rows.keys()):
            if row_idx == -1:
                continue
            row_boxes = sorted(det_rows[row_idx], key=lambda b: b.cx)
            row_color_bgr = tuple(reversed(_get_row_color(_color_idx(row_idx))))
            for i in range(len(row_boxes) - 1):
                c1 = (int(row_boxes[i].cx), int(row_boxes[i].cy))
                c2 = (int(row_boxes[i + 1].cx), int(row_boxes[i + 1].cy))
                cv2.line(img_vis, c1, c2, row_color_bgr, line_thickness)
            for box in row_boxes:
                c = (int(box.cx), int(box.cy))
                cv2.circle(img_vis, c, marker_size, row_color_bgr, -1)
                cv2.circle(img_vis, c, marker_size + 1, (255, 255, 255), 1)

        # --- Text center lines (dashed) per aligned row ---
        # text_rows keys are text_row_idx; color uses text_row_idx directly
        # (consistent with draw_rows which maps det→text for color)
        for text_row_idx in sorted(text_rows.keys()):
            if text_row_idx == -1:
                continue
            row_boxes = sorted(text_rows[text_row_idx], key=lambda b: b.cx)
            line_color_bgr = tuple(reversed(_desaturate_color(_get_row_color(text_row_idx), 0.5)))
            for i in range(len(row_boxes) - 1):
                c1 = (int(row_boxes[i].cx), int(row_boxes[i].cy))
                c2 = (int(row_boxes[i + 1].cx), int(row_boxes[i + 1].cy))
                _draw_dashed_line(img_vis, c1, c2, line_color_bgr, line_thickness, 10)
            for box in row_boxes:
                c = (int(box.cx), int(box.cy))
                cv2.circle(img_vis, c, marker_size - 1, line_color_bgr, -1)

        # --- Labels with PIL ---
        img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_font(24)

        # Detection box labels
        for box in detection_sign_boxes:
            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)
            key = (box.row_idx, box.col_idx)
            info = det_sign_match_info.get(key, {"status": "unmatched", "text_sign_name": None})
            status = info["status"]

            box_height = y2 - y1
            lbl_h = max(int(box_height / 6), 12)
            fsize = max(int(lbl_h * 0.75), 10)
            lbl_font = _get_font(fsize)

            det_label = box.sign_name[:10] if hasattr(box, 'sign_name') else str(box.sign.name[:10])

            if status == "same":
                # Detection label, black bg
                _draw_label(draw, det_label, x1, y1, x2, lbl_h, lbl_font, bg=(0, 0, 0))
            elif status == "diff":
                text_name = info.get("text_sign_name", "")
                # Detection label on top (dark gray bg = detection result)
                _draw_label(draw, det_label, x1, y1, x2, lbl_h, lbl_font, bg=(80, 80, 80))
                # Text label below (colored bg)
                if text_name:
                    color_rgb = self._get_det_sign_color(_color_idx(box.row_idx), "same")
                    _draw_label(draw, text_name[:10], x1, y1 + lbl_h + 1, x2, lbl_h, lbl_font, bg=color_rgb)
            else:
                # Unmatched detection: gray bg
                _draw_label(draw, det_label, x1, y1, x2, lbl_h, lbl_font, bg=(128, 128, 128))

        # Diff-label and unmatched text box labels
        for box in aligned_text_boxes:
            key = (box.row_idx, box.col_idx)
            info = text_sign_match_info.get(key, {"status": "unmatched", "det_sign_name": None})
            if info["status"] == "diff":
                x1, y1 = int(box.x1), int(box.y1)
                x2, y2 = int(box.x2), int(box.y2)
                box_height = y2 - y1
                lbl_h = max(int(box_height / 6), 12)
                fsize = max(int(lbl_h * 0.75), 10)
                lbl_font = _get_font(fsize)
                label = box.sign_name[:10] if hasattr(box, 'sign_name') else str(box.sign.name[:10])
                color_rgb = _desaturate_color(_get_row_color(box.row_idx), 0.4)
                _draw_label(draw, label, x1, y1, x2, lbl_h, lbl_font, bg=color_rgb)
            elif info["status"] == "unmatched":
                x1, y1 = int(box.x1), int(box.y1)
                x2, y2 = int(box.x2), int(box.y2)
                box_height = y2 - y1
                lbl_h = max(int(box_height / 6), 12)
                fsize = max(int(lbl_h * 0.75), 10)
                lbl_font = _get_font(fsize)
                label = box.sign_name[:10] if hasattr(box, 'sign_name') else str(box.sign.name[:10])
                _draw_label(draw, label, x1, y1, x2, lbl_h, lbl_font, bg=(160, 160, 160))

        # --- Row annotations ---
        for row_idx in sorted(det_rows.keys()):
            if row_idx == -1:
                continue
            row_boxes = det_rows[row_idx]
            avg_y = np.mean([b.cy for b in row_boxes])
            display_idx = row_idx + 1
            if row_idx in det_to_text:
                mapped_display = det_to_text[row_idx] + 1
                lbl = f"D{display_idx}→R{mapped_display}"
                color_rgb = _get_row_color(_color_idx(row_idx))
            else:
                lbl = f"D{display_idx}"
                color_rgb = (128, 128, 128)
            draw.text((10, int(avg_y) - 12), lbl, font=font, fill=color_rgb)

        img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.result = img_vis
        return img_vis

    # --- Internal helpers for match-quality coloring ---

    @staticmethod
    def _get_sign_color(row_idx: int, status: str, row_mapping: dict) -> tuple:
        """Get RGB color for a text sign box based on match status."""
        if row_idx not in row_mapping:
            return (128, 128, 128)
        primary = _get_row_color(row_idx)
        if status == "same":
            return primary
        elif status == "diff":
            return _desaturate_color(primary, 0.4)
        else:
            return (128, 128, 128)

    @staticmethod
    def _get_det_sign_color(row_idx: int, status: str) -> tuple:
        """Get RGB color for a detection sign box based on match status."""
        primary = _get_row_color(row_idx)
        if status == "same":
            return primary
        elif status == "diff":
            return _desaturate_color(primary, 0.4)
        else:
            return (128, 128, 128)


# ===== Module-level helper functions =====

def _get_row_color(row_idx: int) -> tuple:
    """Generate distinct RGB color for a row using HSV golden angle."""
    if row_idx == -1:
        return (128, 128, 128)
    hue = int((row_idx * 137.5) % 180)
    hsv_color = np.uint8([[[hue, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))


def _desaturate_color(rgb: tuple, factor: float = 0.4) -> tuple:
    """Reduce saturation of an RGB color. factor=0 → grayscale, factor=1 → original."""
    r, g, b = rgb
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv[1] = int(hsv[1] * factor)
    bgr = cv2.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _get_font(size: int):
    """Load DejaVuSans font at given size, with fallback."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_label(draw: ImageDraw.Draw, text: str, x1: int, y_top: int, x2: int,
                label_height: int, font, bg: tuple = (0, 0, 0), fg: tuple = (255, 255, 255)):
    """Draw a text label with background at specified position."""
    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    lx1, ly1 = x1 + 2, y_top + 2
    lx2, ly2 = min(x1 + tw + 8, x2 - 2), y_top + label_height
    draw.rectangle([lx1, ly1, lx2, ly2], fill=bg)
    text_y = ly1 + (label_height - th) // 2
    draw.text((lx1 + 4, text_y), text, font=font, fill=fg)


def _draw_dashed_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = 2, dash_length: int = 8):
    """Draw a dashed rectangle on an OpenCV image (BGR color)."""
    x1, y1 = pt1
    x2, y2 = pt2
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)
    _draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness, dash_length)
    _draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness, dash_length)


def _draw_dashed_line(img: np.ndarray, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = 2, dash_length: int = 10):
    """Draw a dashed line between two points on an OpenCV image (BGR color)."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    pos = 0.0
    drawing = True
    while pos < dist:
        end = min(pos + dash_length, dist)
        sx = int(x1 + dx * pos)
        sy = int(y1 + dy * pos)
        ex = int(x1 + dx * end)
        ey = int(y1 + dy * end)
        if drawing:
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos = end + dash_length / 2 if drawing else end + dash_length / 2
        drawing = not drawing


def build_sign_match_info(row_sign_matches: dict, text_to_det: dict,
                          det_rows_dict: dict, optim_sign_boxes: List):
    """
    Build sign_match_info dicts for text and detection perspectives.

    Args:
        row_sign_matches: {text_row_idx: [(text_sign_idx, det_sign_idx), ...]}
        text_to_det: {text_row_idx: det_row_idx}
        det_rows_dict: {det_row_idx: [SignBox, ...]} from sub_tablet_detection.get_rows_dict()
        optim_sign_boxes: List of SignBox from sub_tablet_optim

    Returns:
        text_sign_match_info: {(text_row_idx, text_col_idx): {"status", "det_sign_name"}}
        det_sign_match_info:  {(det_row_idx, det_col_idx): {"status", "text_sign_name"}}
    """
    # Build match pairs
    match_pairs = {}       # (text_row_idx, text_sign_idx) → (det_row_idx, det_sign_idx)
    matched_det_keys = set()

    for text_row_idx, sign_matches in row_sign_matches.items():
        det_row_idx = text_to_det[text_row_idx]
        for t_idx, d_idx in sign_matches:
            match_pairs[(text_row_idx, t_idx)] = (det_row_idx, d_idx)
            matched_det_keys.add((det_row_idx, d_idx))

    text_info = {}
    for sb in optim_sign_boxes:
        key = (sb.row_idx, sb.col_idx)
        if key in match_pairs:
            det_row_idx, det_sign_idx = match_pairs[key]
            det_box = det_rows_dict[det_row_idx][det_sign_idx]
            det_name = det_box.sign_name if hasattr(det_box, 'sign_name') else det_box.sign.name
            text_name = sb.sign_name if hasattr(sb, 'sign_name') else sb.sign.name
            if text_name == det_name:
                text_info[key] = {"status": "same", "det_sign_name": det_name}
            else:
                text_info[key] = {"status": "diff", "det_sign_name": det_name}
        else:
            text_info[key] = {"status": "unmatched", "det_sign_name": None}

    # Build reverse lookup: det_key → text_key for O(1) access
    det_to_text_key = {det_key: text_key for text_key, det_key in match_pairs.items()}
    # Build text box lookup: (row_idx, col_idx) → SignBox
    text_box_lookup = {(sb.row_idx, sb.col_idx): sb for sb in optim_sign_boxes}

    det_info = {}
    for det_row_idx, det_row_boxes in det_rows_dict.items():
        for d_idx, det_box in enumerate(det_row_boxes):
            dk = (det_row_idx, d_idx)
            det_name = det_box.sign_name if hasattr(det_box, 'sign_name') else det_box.sign.name
            if dk in det_to_text_key:
                text_key = det_to_text_key[dk]
                text_box = text_box_lookup.get(text_key)
                if text_box:
                    text_name = text_box.sign_name if hasattr(text_box, 'sign_name') else text_box.sign.name
                    if det_name == text_name:
                        det_info[dk] = {"status": "same", "text_sign_name": text_name}
                    else:
                        det_info[dk] = {"status": "diff", "text_sign_name": text_name}
                else:
                    det_info[dk] = {"status": "unmatched", "text_sign_name": None}
            else:
                det_info[dk] = {"status": "unmatched", "text_sign_name": None}

    return text_info, det_info


class CompositeVisualizer:
    """Compose multiple images into a grid layout with optional titles."""

    def __init__(self):
        self.result = None

    def compose(self, images: List[np.ndarray], layout: tuple,
                titles: List[str] = None, figsize: tuple = None,
                title_height: int = 40, padding: int = 4) -> np.ndarray:
        """
        Compose multiple BGR images into a grid at full original resolution.

        Images are resized so that each row has uniform height and each column
        has uniform width, preserving the original pixel data as much as
        possible.  Titles are rendered with PIL for Unicode support.

        Args:
            images: List of BGR images (from .result attributes)
            layout: (rows, cols) grid layout
            titles: Optional list of title strings for each image
            figsize: Ignored (kept for API compatibility)
            title_height: Pixel height of the title bar above each cell
            padding: Pixel gap between cells

        Returns:
            Composed BGR image
        """
        n_rows, n_cols = layout
        t_h = title_height if titles else 0

        # --- collect cells (pad missing slots with None) ---
        cells: List[Optional[np.ndarray]] = []
        for idx in range(n_rows * n_cols):
            if idx < len(images) and images[idx] is not None:
                cells.append(images[idx])
            else:
                cells.append(None)

        # --- determine uniform cell size per row/col ---
        # max height per row, max width per column (from original images)
        row_heights = [0] * n_rows
        col_widths = [0] * n_cols
        for idx, cell in enumerate(cells):
            r, c = divmod(idx, n_cols)
            if cell is not None:
                h, w = cell.shape[:2]
                row_heights[r] = max(row_heights[r], h)
                col_widths[c] = max(col_widths[c], w)

        # fallback for completely empty rows/cols
        default_h = max(row_heights) if any(row_heights) else 100
        default_w = max(col_widths) if any(col_widths) else 100
        row_heights = [h if h > 0 else default_h for h in row_heights]
        col_widths = [w if w > 0 else default_w for w in col_widths]

        # --- build per-cell images (resize + title bar) ---
        def _make_cell(cell_img: Optional[np.ndarray], cell_h: int, cell_w: int,
                       title: Optional[str]) -> np.ndarray:
            bar = np.ones((t_h, cell_w, 3), dtype=np.uint8) * 255 if t_h > 0 else None

            if cell_img is None:
                canvas = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 220
            else:
                ch, cw = cell_img.shape[:2]
                if ch == cell_h and cw == cell_w:
                    canvas = cell_img.copy()
                else:
                    # scale keeping aspect ratio, center on white canvas
                    scale = min(cell_w / cw, cell_h / ch)
                    new_w, new_h = int(cw * scale), int(ch * scale)
                    resized = cv2.resize(cell_img, (new_w, new_h),
                                         interpolation=cv2.INTER_AREA)
                    canvas = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 220
                    y0 = (cell_h - new_h) // 2
                    x0 = (cell_w - new_w) // 2
                    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

            if bar is not None:
                if title:
                    bar_pil = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(bar_pil)
                    font = _get_font(max(t_h - 12, 14))
                    bbox = draw.textbbox((0, 0), title, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = (cell_w - tw) // 2
                    ty = (t_h - th) // 2
                    draw.text((tx, ty), title, font=font, fill=(0, 0, 0))
                    bar = cv2.cvtColor(np.array(bar_pil), cv2.COLOR_RGB2BGR)
                return np.vstack([bar, canvas])
            return canvas

        # --- assemble grid ---
        grid_rows = []
        for r in range(n_rows):
            row_cells = []
            for c in range(n_cols):
                idx = r * n_cols + c
                title = titles[idx] if titles and idx < len(titles) else None
                cell_img = _make_cell(cells[idx], row_heights[r], col_widths[c], title)
                row_cells.append(cell_img)
            # add horizontal padding between columns
            if padding > 0 and len(row_cells) > 1:
                pad_h = row_cells[0].shape[0]
                pad_col = np.ones((pad_h, padding, 3), dtype=np.uint8) * 255
                merged = [row_cells[0]]
                for rc in row_cells[1:]:
                    merged.append(pad_col)
                    merged.append(rc)
                row_img = np.hstack(merged)
            else:
                row_img = np.hstack(row_cells)
            grid_rows.append(row_img)

        # add vertical padding between rows
        if padding > 0 and len(grid_rows) > 1:
            total_w = grid_rows[0].shape[1]
            pad_row = np.ones((padding, total_w, 3), dtype=np.uint8) * 255
            merged = [grid_rows[0]]
            for gr in grid_rows[1:]:
                # ensure same width (can differ by 1 px due to rounding)
                if gr.shape[1] != total_w:
                    gr = cv2.resize(gr, (total_w, gr.shape[0]))
                merged.append(pad_row)
                merged.append(gr)
            self.result = np.vstack(merged)
        else:
            self.result = np.vstack(grid_rows)

        return self.result

    def display_result(self, vis_opt: str = "draw", path: str = None):
        """Display or save the composed image."""
        if self.result is None:
            print("No result. Call compose() first.")
            return
        if vis_opt == "draw":
            plt.figure(figsize=(16, 10))
            plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        elif vis_opt == "save":
            if path is None:
                path = 'composite.jpg'
            import os
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            cv2.imwrite(path, self.result)
            print(f"✓ Saved to: {os.path.abspath(path)}")

    def save(self, path: str):
        """Save to file."""
        self.display_result(vis_opt="save", path=path)

    def show_draw(self):
        """Display using matplotlib."""
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
