import mediapipe as mp
import cv2
import os
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_image(img_path, save_path, hands):
    img = cv2.imread(img_path)
    if img is None:
        return
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, hand_lms,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2)
            )
    cv2.imwrite(save_path, image_bgr)

def process_folder(folder_path, hands):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                save_dir = root.replace('dataset', 'hand_landmark_dataset')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file)
                process_image(img_path, save_path, hands)
                print(f'Processed: {img_path} -> {save_path}')

if __name__ == "__main__":
    base_dir = "tensorFlow/dataset"
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        process_folder(os.path.join(base_dir, "train"), hands)
        process_folder(os.path.join(base_dir, "test"), hands)