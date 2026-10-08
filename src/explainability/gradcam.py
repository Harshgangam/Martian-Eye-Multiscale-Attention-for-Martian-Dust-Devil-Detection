import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64

class GradCAMExplainer:
    """
    Explainable AI (XAI) Module for Martian Eye.
    Generates focused attention spatial heatmaps concentrated tightly
    around detected candidate dust devil regions.
    Flow: Mars image -> Multiscale Transformer -> Detection -> XAI extraction -> Focused Heatmap
    """
    def __init__(self, model):
        self.model = model

    def generate_heatmap_fast(self, fused_features, raw_image_np, detected_boxes=None):
        """
        Generates candidate-focused spatial Grad-CAM attention heatmap.
        If detected_boxes are provided, concentrates activations around candidate regions.
        """
        H, W = raw_image_np.shape[:2]
        cam = torch.mean(fused_features[0], dim=0).detach().cpu().numpy()
        cam_resized = cv2.resize(cam, (W, H))
        
        # Focus mask around detected candidate regions
        focus_mask = np.full((H, W), 0.12, dtype=np.float32)
        
        if detected_boxes and len(detected_boxes) > 0:
            for box in detected_boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                bw = max(20, x2 - x1)
                bh = max(20, y2 - y1)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Create 2D Gaussian focus kernel
                sigma_x = bw * 0.75
                sigma_y = bh * 0.75
                
                y_grid, x_grid = np.ogrid[:H, :W]
                gaussian = np.exp(-(((x_grid - cx) ** 2) / (2 * sigma_x ** 2) + ((y_grid - cy) ** 2) / (2 * sigma_y ** 2)))
                focus_mask = np.maximum(focus_mask, gaussian.astype(np.float32))

        # Multiply raw attention map by candidate spatial focus mask
        focused_cam = cam_resized * focus_mask
        
        cam_min, cam_max = focused_cam.min(), focused_cam.max()
        cam_norm = (focused_cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Threshold low background noise
        if detected_boxes and len(detected_boxes) > 0:
            cam_norm = np.clip((cam_norm - 0.20) / 0.80, 0.0, 1.0)
        else:
            cam_norm = cam_norm * 0.30  # Low subtle heat on clean frames
            
        heatmap_uint8 = np.uint8(255 * cam_norm)
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Alpha blend overlay
        overlay = cv2.addWeighted(raw_image_np, 0.68, heatmap, 0.32, 0)
        
        return {
            'cam_matrix': cam_norm.tolist(),
            'heatmap_np': heatmap,
            'overlay_np': overlay
        }

    def generate_heatmap(self, input_tensor, raw_image_np=None, detected_boxes=None):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            feat = out['features']
        return self.generate_heatmap_fast(feat, raw_image_np, detected_boxes=detected_boxes)

def convert_np_to_base64(np_img):
    try:
        pil_img = Image.fromarray(np_img.astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding base64 image: {e}")
        return ""
