# In your project root (same folder as jet_annotator_app.py), create a file visualize_mask.py:

import cv2
import numpy as np
from jet_inference import JetSegmenter

# Open any video from your data folder
cap = cv2.VideoCapture("MEW-dalton.mp4")  # or your pulsing video
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # jump to frame 100
ret, frame = cap.read()
cap.release()

if ret:
    print(f"Frame: {frame.shape}")
    
    seg = JetSegmenter("checkpoints/unet_mobilenet.onnx", confidence=0.9)
    mask = seg.predict_mask(frame)
    skeleton = seg.extract_skeleton(mask)
    
    print(f"Mask: {mask.shape}, {np.sum(mask)} jet pixels")
    print(f"Skeleton: {len(skeleton) if skeleton else 0} points")
    
    # Visualize three panels
    mask_vis = (mask * 255).astype(np.uint8)
    mask_3ch = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    
    # Draw skeleton on top of mask
    mask_with_skel = mask_3ch.copy()
    if skeleton:
        skeleton = np.array(skeleton, dtype=np.int32)
        cv2.polylines(mask_with_skel, [skeleton], False, (0, 255, 0), 2)
    
    # Overlay mask on original frame
    overlay = cv2.addWeighted(frame, 0.6, mask_3ch, 0.4, 0)
    
    # Stack side-by-side for comparison
    top = np.hstack([frame, mask_3ch])
    bottom = np.hstack([overlay, mask_with_skel])
    combined = np.vstack([top, bottom])
    
    cv2.imshow("Top-left: Raw frame | Top-right: Binary mask | Bottom-left: Overlay | Bottom-right: Mask+skeleton", combined)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()