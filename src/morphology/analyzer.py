import numpy as np
import cv2

class DustDevilMorphologyAnalyzer:
    """
    Novelty 8: Morphology Analysis.
    Extracts measurable image-supported morphological parameters of detected 
    dust-devil atmospheric columns and plumes.
    """
    @staticmethod
    def analyze_bbox(bbox, image_shape=(512, 512), pixel_scale_m_per_px=0.5):
        """
        bbox: [x1, y1, x2, y2] normalized (0 to 1) or in absolute pixel coordinates.
        image_shape: (H, W)
        """
        H, W = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Denormalize if normalized
        if x2 <= 1.0 and y2 <= 1.0:
            x1_px, y1_px = int(x1 * W), int(y1 * H)
            x2_px, y2_px = int(x2 * W), int(y2 * H)
        else:
            x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)
            
        width_px = max(1, x2_px - x1_px)
        height_px = max(1, y2_px - y1_px)
        area_px = width_px * height_px
        aspect_ratio = round(width_px / height_px, 3)
        
        # Atmospheric column orientation angle (degrees relative to vertical)
        # Vertical columnar dust devils typically have aspect ratio < 1 (taller than wide)
        orientation_deg = round(np.arctan2(width_px, height_px) * (180.0 / np.pi), 1)
        
        # Shape compactness (4 * pi * area / perimeter^2)
        perimeter_px = 2 * (width_px + height_px)
        compactness = round((4 * np.pi * area_px) / (perimeter_px ** 2), 3)
        
        # Physical estimation (assuming approximate spatial rover/orbital calibration)
        est_height_meters = round(height_px * pixel_scale_m_per_px, 2)
        est_diameter_meters = round(width_px * pixel_scale_m_per_px, 2)
        
        # Morphology Classification
        if aspect_ratio < 0.4:
            column_type = "Slender Vertical Column"
        elif aspect_ratio < 0.8:
            column_type = "Standard Funnel Column"
        elif aspect_ratio <= 1.2:
            column_type = "Broad Diffuse Plume"
        else:
            column_type = "Low Horizontal Dust Skirt"
            
        return {
            'bbox_px': [x1_px, y1_px, x2_px, y2_px],
            'width_px': width_px,
            'height_px': height_px,
            'area_px': area_px,
            'aspect_ratio': aspect_ratio,
            'orientation_deg': orientation_deg,
            'compactness': compactness,
            'est_height_m': est_height_meters,
            'est_diameter_m': est_diameter_meters,
            'column_type': column_type,
            'units': 'Pixels (Image-based) + Estimated Meters (Calibrated Spatial Proxy)'
        }
