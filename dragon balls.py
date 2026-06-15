import cv2
import mediapipe as mp
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def get_angle_3d(a, b, c):
    v1 = [a.x - b.x, a.y - b.y, a.z - b.z]
    v2 = [c.x - b.x, c.y - b.y, c.z - b.z]
    
    dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
        
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle)) 
    
    return math.degrees(math.acos(cos_angle))

def get_looped_frame(cap):
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap_energy = cv2.VideoCapture('assets/energy.mp4')
    cap_kame = cv2.VideoCapture('assets/kamehameha.mp4')
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    landmarker = PoseLandmarker.create_from_options(options)
    prev_state = "IDLE"
    frame_idx = 0
    
    smooth_hx, smooth_hy = -1, -1
    smooth_factor = 0.5 
    idle_frames = 0
    
    with landmarker as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image = cv2.flip(image, 1)
            image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
            
            height, width, _ = image.shape
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            frame_idx += 1
            frame_timestamp_ms = frame_idx * 33
            pose_landmarker_result = pose.detect_for_video(mp_image, frame_timestamp_ms)
            
            current_state = "IDLE"
            hx, hy = -1, -1 
            
            if pose_landmarker_result.pose_landmarks:
                for pose_landmarks in pose_landmarker_result.pose_landmarks:
                    left_shoulder = pose_landmarks[11]
                    right_shoulder = pose_landmarks[12]
                    left_elbow = pose_landmarks[13]
                    right_elbow = pose_landmarks[14]
                    left_wrist = pose_landmarks[15]
                    right_wrist = pose_landmarks[16]
                    
                    if left_wrist.visibility > 0.4 or right_wrist.visibility > 0.4:
                        wrist_dist = math.hypot(left_wrist.x - right_wrist.x, left_wrist.y - right_wrist.y)
                        dist_thresh = 0.5 if prev_state == "FIRING" else 0.35
                        
                        if wrist_dist < dist_thresh:
                            left_angle = get_angle_3d(left_shoulder, left_elbow, left_wrist)
                            right_angle = get_angle_3d(right_shoulder, right_elbow, right_wrist)
                            angle_thresh = 120 if prev_state == "FIRING" else 135
                            
                            if max(left_angle, right_angle) > angle_thresh:
                                current_state = "FIRING"
                                if left_wrist.visibility > right_wrist.visibility:
                                    hx = int(left_wrist.x * width)
                                    hy = int(left_wrist.y * height)
                                else:
                                    hx = int(right_wrist.x * width)
                                    hy = int(right_wrist.y * height)
                            else:
                                current_state = "CHARGING"
                                hx = int(((left_wrist.x + right_wrist.x)/2) * width)
                                hy = int(((left_wrist.y + right_wrist.y)/2) * height)
            
            if current_state == "IDLE" and prev_state != "IDLE":
                idle_frames += 1
                if idle_frames < 5:
                    current_state = prev_state
            else:
                idle_frames = 0
            
            if hx != -1:
                if smooth_hx == -1: 
                    smooth_hx, smooth_hy = hx, hy
                else:
                    smooth_hx = int((1 - smooth_factor) * smooth_hx + smooth_factor * hx)
                    smooth_hy = int((1 - smooth_factor) * smooth_hy + smooth_factor * hy)
            elif current_state == "IDLE":
                smooth_hx, smooth_hy = -1, -1
            
            if current_state == "CHARGING" and prev_state != "CHARGING":
                cap_energy.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if current_state == "FIRING" and prev_state != "FIRING":
                cap_kame.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            prev_state = current_state
            
            if current_state == "CHARGING" and smooth_hx != -1:
                ef = get_looped_frame(cap_energy)
                if ef is not None:
                    size = int(height * 0.35)
                    ef_resized = cv2.resize(ef, (size, size))
                    
                    x1 = max(0, smooth_hx - size//2)
                    y1 = max(0, smooth_hy - size//2)
                    x2 = min(width, smooth_hx + size//2)
                    y2 = min(height, smooth_hy + size//2)
                    
                    ex1 = size//2 - (smooth_hx - x1)
                    ey1 = size//2 - (smooth_hy - y1)
                    ex2 = size//2 + (x2 - smooth_hx)
                    ey2 = size//2 + (y2 - smooth_hy)
                    
                    if x2 > x1 and y2 > y1:
                        roi = image[y1:y2, x1:x2]
                        effect_roi = ef_resized[ey1:ey2, ex1:ex2]
                        image[y1:y2, x1:x2] = cv2.add(roi, effect_roi)
                        
            elif current_state == "FIRING" and smooth_hx != -1:
                ef = get_looped_frame(cap_kame)
                if ef is not None:
                    orig_h, orig_w = ef.shape[:2]
                    
                    beam_h = int(height * 0.75) 
                    beam_w = int(beam_h * (orig_w / orig_h))
                    ef_resized = cv2.resize(ef, (beam_w, beam_h))
                    
                    offset_x = beam_w // 2 
                    offset_y = 0
                    
                    target_x = smooth_hx + offset_x
                    target_y = smooth_hy + offset_y
                    
                    x1 = max(0, target_x - beam_w//2)
                    y1 = max(0, target_y - beam_h//2)
                    x2 = min(width, target_x + beam_w//2)
                    y2 = min(height, target_y + beam_h//2)
                    
                    ex1 = beam_w//2 - (target_x - x1)
                    ey1 = beam_h//2 - (target_y - y1)
                    ex2 = beam_w//2 + (x2 - target_x)
                    ey2 = beam_h//2 + (y2 - target_y)
                    
                    if x2 > x1 and y2 > y1:
                        roi = image[y1:y2, x1:x2]
                        effect_roi = ef_resized[ey1:ey2, ex1:ex2]
                        image[y1:y2, x1:x2] = cv2.add(roi, effect_roi)
            
            cv2.imshow('dragon ball z', image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cap_energy.release()
    cap_kame.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    