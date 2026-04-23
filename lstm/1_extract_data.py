"""
Step 1: Extract and preprocess data for 'good' and 'funny' signs
This script processes only the selected classes and creates train/val/test splits
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = os.path.join(os.getcwd(), 'videosDataset')
SELECTED_CLASSES = None  # Auto-discover classes from videosDataset when None
SEQ_LENGTH = 30
FEATURE_VECTOR_SIZE = 1662

# Output files
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Setup
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    """Processes an image with the MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extracts and normalizes keypoints from MediaPipe results.
    Returns a 1662-feature vector or zeros if no pose detected.
    """
    if not results.pose_landmarks:
        return np.zeros(FEATURE_VECTOR_SIZE)

    pose_lm = results.pose_landmarks.landmark
    
    # Anchor Point (nose)
    anchor = np.array([pose_lm[0].x, pose_lm[0].y, pose_lm[0].z])
    
    # Scale Factor (shoulder width)
    s_left = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
    s_right = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
    scale = np.linalg.norm(s_left - s_right)

    if scale < 1e-6:
        return np.zeros(FEATURE_VECTOR_SIZE)

    # Extract and normalize POSE (33 landmarks)
    pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm])
    pose_visibility = np.array([[lm.visibility] for lm in pose_lm])
    normalized_pose_coords = (pose_coords - anchor) / scale
    pose = np.concatenate([normalized_pose_coords, pose_visibility], axis=1).flatten()

    # Extract and normalize FACE (468 landmarks)
    if results.face_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        normalized_face_coords = (face_coords - anchor) / scale
        face = normalized_face_coords.flatten()
    else:
        face = np.zeros(468 * 3)

    # Extract and normalize LEFT HAND (21 landmarks)
    if results.left_hand_landmarks:
        lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        normalized_lh_coords = (lh_coords - anchor) / scale
        left_hand = normalized_lh_coords.flatten()
    else:
        left_hand = np.zeros(21 * 3)

    # Extract and normalize RIGHT HAND (21 landmarks)
    if results.right_hand_landmarks:
        rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        normalized_rh_coords = (rh_coords - anchor) / scale
        right_hand = normalized_rh_coords.flatten()
    else:
        right_hand = np.zeros(21 * 3)

    return np.concatenate([pose, face, left_hand, right_hand])

def sample_frames(video_frames, seq_length):
    """Samples or pads frames to fixed sequence length."""
    if len(video_frames) > seq_length:
        indices = np.linspace(0, len(video_frames) - 1, seq_length, dtype=int)
        sampled_frames = [video_frames[i] for i in indices]
    else:
        sampled_frames = video_frames
        pad_len = seq_length - len(video_frames)
        for _ in range(pad_len):
            sampled_frames.append(np.zeros(FEATURE_VECTOR_SIZE))
    return np.array(sampled_frames)

def process_data():
    """Main data processing function."""
    if not os.path.isdir(DATA_PATH):
        print(f"❌ Error: Dataset path not found: {DATA_PATH}")
        print("Create videosDataset and add one folder per word with .mp4 files.")
        return

    # Auto-discover classes with at least one mp4 video, or use explicitly selected classes.
    classes = SELECTED_CLASSES if SELECTED_CLASSES else sorted(
        [
            folder for folder in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, folder))
            and any(f.lower().endswith('.mp4') for f in os.listdir(os.path.join(DATA_PATH, folder)))
        ]
    )

    if not classes:
        print(f"❌ Error: No class folders with .mp4 files found in {DATA_PATH}")
        return

    print("="*60)
    print("SIGN LANGUAGE DATA EXTRACTION - PROFESSIONAL PIPELINE")
    print("="*60)
    print(f"\nProcessing classes: {classes}")
    
    # Create class mapping
    class_map = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Save class labels
    with open(os.path.join(OUTPUT_DIR, 'classes.json'), 'w') as f:
        json.dump(classes, f)
    print(f"✓ Saved class labels to {OUTPUT_DIR}/classes.json")
    
    sequences, labels, video_names = [], [], []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for class_name in classes:
            class_index = class_map[class_name]
            class_path = os.path.join(DATA_PATH, class_name)
            
            if not os.path.isdir(class_path):
                print(f"⚠ Warning: {class_path} not found")
                continue
            
            videos = [v for v in os.listdir(class_path) if v.lower().endswith('.mp4')]
            print(f"\n📹 Processing '{class_name}': {len(videos)} videos")
            
            for video_name in tqdm(videos, desc=f"  {class_name}"):
                video_path = os.path.join(class_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    print(f"  ⚠ Could not open: {video_name}")
                    continue
                
                video_frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    video_frames.append(keypoints)
                
                cap.release()
                
                if not video_frames:
                    print(f"  ⚠ No frames from: {video_name}")
                    continue
                
                sequence = sample_frames(video_frames, SEQ_LENGTH)
                sequences.append(sequence)
                labels.append(class_index)
                video_names.append(f"{class_name}_{video_name}")
    
    print(f"\n✓ Total sequences processed: {len(sequences)}")
    
    if not sequences:
        print("❌ Error: No sequences were processed!")
        return
    
    # Convert to arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Shape of X: {X.shape}")
    print(f"  - Shape of y: {y.shape}")
    print(f"  - Classes: {classes}")
    for i, class_name in enumerate(classes):
        count = np.sum(y == i)
        print(f"    • {class_name}: {count} samples")
    
    # Train/Val/Test split with stratification
    # 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📂 Data Splits:")
    print(f"  - Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Save all splits
    np.savez(
        os.path.join(OUTPUT_DIR, 'dataset.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    print(f"\n✅ Successfully saved processed data to {OUTPUT_DIR}/dataset.npz")
    print("="*60)

if __name__ == "__main__":
    process_data()
