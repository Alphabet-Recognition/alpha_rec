import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import sys
import random
import string
import time

class HandGUI:
    def __init__(self, model=None, class_names=None):
        self.model = model
        self.class_names = class_names or []
        self.img_height, self.img_width = 160, 160

        # Prediction buffer (like typed text)
        self.global_variable = ""

        # Random practice list (20 uppercase letters) - initialized but generation
        # will happen after the bottom overlay / labels are created so the
        # target_label exists when update_target_label() runs.
        self.random_list = []
        # timestamp until which model calls / detection should be paused
        # (seconds since epoch). 0 means no pause.
        self._paused_until = 0.0

        # Performance settings
        self.DETECT_WIDTH = 640
        self.PROCESS_EVERY_N = 2
        self.FRAME_DELAY_MS = 15

        # Hand detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.5)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: could not open camera", file=sys.stderr)
            sys.exit(1)

        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Hand Tracking + CNN")

        # Canvas for video
        self.canvas = tk.Canvas(self.root, width=self.camera_width, height=self.camera_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.image = None

        # Bottom overlay
        self.bottom_frame = tk.Frame(self.canvas, bd=1, relief=tk.FLAT)
        self._bottom_window_id = self.canvas.create_window(0, 0, anchor='s', window=self.bottom_frame)
        # Target label shows current letter to practice
        # create the target_label first; its text will be updated immediately
        # after random list generation.
        self.target_label = tk.Label(
            self.bottom_frame,
            text="Target: --",
            font=("TkDefaultFont", 12, "bold"),
            anchor="w", justify="left"
        )
        self.target_label.pack(side=tk.LEFT, padx=(6, 6), pady=6)
        self.variable_label = tk.Label(
            self.bottom_frame,
            text="Waiting for detection...",
            font=("TkDefaultFont", 12),
            anchor="w", justify="left"
        )
        self.variable_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bottom_frame.configure(height=42)

        # Button to regenerate the random 20-letter practice list
        self.random_btn = tk.Button(self.bottom_frame, text="Random Again", command=self.on_random_again)
        self.random_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Now that target_label exists, generate the random list and update UI.
        self.generate_random_list()

        self.canvas.bind("<Configure>", self._reposition_bottom_overlay)
        self.root.after(50, self._reposition_bottom_overlay)

        # Frame processing state
        self.current_frame_bgr = None
        self.latest_hands = []
        self._frame_counter = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _reposition_bottom_overlay(self, event=None):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        margin_y = 12
        self.canvas.coords(self._bottom_window_id, cw // 2, ch - margin_y)
        new_w = max(60, cw - 24)
        self.bottom_frame.configure(width=new_w)
        self.variable_label.configure(wraplength=max(20, new_w - 80))

    def generate_random_list(self, length=20):
        """Generate a random list of unique uppercase letters (length <= 26)."""
        length = min(max(1, length), 26)
        letters = list(string.ascii_uppercase)
        self.random_list = random.sample(letters, k=length)
        self.update_target_label()

    def update_target_label(self):
        if self.random_list:
            self.target_label.config(text="Target: " + self.random_list[0] + "->" + str(self.random_list[1:]))
        else:
            self.target_label.config(text="Target: Done")

    def show_temporary_popup(self, message, duration_ms=5000):
        """Show a small undecorated popup centered over the canvas for duration_ms."""
        # pause predictions while popup is visible
        try:
            self._paused_until = time.time() + (duration_ms / 1000.0)
        except Exception:
            self._paused_until = time.time() + (duration_ms / 1000.0)
        try:
            popup = tk.Toplevel(self.root)
            popup.overrideredirect(True)
            lbl = tk.Label(popup, text=message, bg="#28a745", fg="white", font=("TkDefaultFont", 14, "bold"), bd=6)
            lbl.pack()

            # Position popup centered over the canvas
            self.root.update_idletasks()
            canvas_x = self.canvas.winfo_rootx()
            canvas_y = self.canvas.winfo_rooty()
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            popup.update_idletasks()
            pw = popup.winfo_width()
            ph = popup.winfo_height()
            x = canvas_x + (canvas_w - pw) // 2
            y = canvas_y + (canvas_h - ph) // 2
            popup.geometry(f"+{x}+{y}")
            popup.lift()
            popup.after(duration_ms, popup.destroy)
        except Exception:
            # Fail silently if popup can't be shown
            pass

    def on_random_again(self):
        self.generate_random_list()
        # reset typed buffer so user starts fresh
        self.global_variable = ""
        self.variable_label.config(text=self.global_variable or "Waiting for detection...")

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def is_hand_fully_visible(self, hand, frame, margin=30, threshold=0.85):
        h, w, _ = frame.shape
        x, y, w_box, h_box = hand['bbox']

        # Compute hand box area
        hand_area = w_box * h_box

        # Compute visible region within frame boundaries
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + w_box + margin)
        y2 = min(h, y + h_box + margin)

        visible_width = max(0, x2 - x1)
        visible_height = max(0, y2 - y1)
        visible_area = visible_width * visible_height

        # Compute visibility ratio
        visible_ratio = visible_area / float(hand_area + 1e-5)

        return visible_ratio >= threshold

    def crop_hand_square(self, frame, hand, margin=130):
        lmList = hand["lmList"]

        xs = [lm[0] for lm in lmList]
        ys = [lm[1] for lm in lmList]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.img_width, self.img_height))
        return crop

    def predict_hand(self, frame, hand, margin=30):
        lmList = hand["lmList"]

        if not self.is_hand_fully_visible(hand, frame):
            print("⚠️ Hand not fully visible, skipping prediction.")
            return None

        if not lmList:
            return None

        xs = [lm[0] for lm in lmList]
        ys = [lm[1] for lm in lmList]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        
        crop = cv2.resize(crop, (self.img_width, self.img_height))
        
        debug_preview = cv2.resize(crop, (self.img_width, self.img_height)) 
        cv2.imshow("Hand Crop", debug_preview)
        
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype("float64") / 255.0
        crop = np.expand_dims(crop, axis=0)

        prediction = self.model.predict(crop)
        pred_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        print("Predicted index:", pred_index, "Confidence:", confidence)

        if self.class_names:
            print("Predicted class name:", self.class_names[pred_index])

        return pred_index, confidence
    
    def apply_prediction(self, pred_class):
        pred_class_str = str(pred_class).strip()

        # Special tokens
        if pred_class_str == "del":
            # clear buffer (user requested delete)
            self.global_variable = ""
        elif pred_class_str == "space":
            # keep single space as the current buffer
            self.global_variable = " "
        elif pred_class_str == "nothing":
            # ignore (do not change buffer)
            pass
        else:
            # For normal predictions, keep only the latest predicted string.
            self.global_variable = pred_class_str

        # Update typed buffer label
        self.variable_label.config(text=self.global_variable or "Waiting for detection...")

        # If the current buffer is a single alphabet letter and matches the
        # current target, show popup, advance (pop) the random list and clear buffer.
        if self.random_list and self.global_variable:
            buf = self.global_variable.strip()
            if len(buf) == 1 and buf.isalpha():
                if buf.upper() == self.random_list[0].upper():
                    # show short popup indicating correct match
                    try:
                        self.show_temporary_popup(f"Correct: {buf.upper()}", duration_ms=500)
                    except Exception:
                        pass
                    # remove target and reset buffer
                    self.random_list.pop(0)
                    self.global_variable = ""
                    self.variable_label.config(text="Waiting for detection...")
                    self.update_target_label()

    def update_frame(self):
        self._frame_counter += 1
        success, frame = self.cap.read()
        if not success or frame is None or frame.size == 0:
            self.root.after(self.FRAME_DELAY_MS, self.update_frame)
            return

        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        frame = cv2.flip(frame, 1)

        # Resize for hand detection
        try:
            h, w = frame.shape[:2]
            scale = self.DETECT_WIDTH / float(w) if w > self.DETECT_WIDTH else 1.0
            small_w = max(1, int(w * scale))
            small_h = max(1, int(h * scale))
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            small = frame
            small_h, small_w = frame.shape[:2]

        # Process every N frames, but skip detection/prediction while paused.
        if time.time() >= self._paused_until:
            if (self._frame_counter % self.PROCESS_EVERY_N) == 0:
                try:
                    hands, _ = self.detector.findHands(small, draw=False)
                    if hands:
                        # scale detected coordinates from small -> original frame
                        scale_x = float(w) / float(small_w)
                        scale_y = float(h) / float(small_h)
                        scaled_hands = []
                        for hand in hands:
                            # scale lmList
                            lm = hand.get("lmList", [])
                            scaled_lm = [[int(pt[0] * scale_x), int(pt[1] * scale_y), pt[2] if len(pt) > 2 else 0] for pt in lm]
                            # scale bbox (x, y, w, h)
                            bx, by, bw_box, bh_box = hand.get("bbox", (0, 0, 0, 0))
                            bx = int(bx * scale_x)
                            by = int(by * scale_y)
                            bw_box = int(bw_box * scale_x)
                            bh_box = int(bh_box * scale_y)
                            scaled_hand = hand.copy()
                            scaled_hand["lmList"] = scaled_lm
                            scaled_hand["bbox"] = (bx, by, bw_box, bh_box)
                            scaled_hands.append(scaled_hand)
                        self.latest_hands = scaled_hands
                    else:
                        self.latest_hands = []
                except Exception:
                    self.latest_hands = []

                # Predict hand gesture using scaled coordinates on original frame
                if self.latest_hands:
                    result = self.predict_hand(frame, self.latest_hands[0])
                    if result:
                        pred_index, conf = result
                        if 0 <= pred_index < len(self.class_names):
                            pred_class_name = self.class_names[pred_index]
                        else:
                            pred_class_name = str(pred_index)
                        self.apply_prediction(pred_class_name)
        else:
            # During pause period, do not call detector/model.
            pass

        # latest_hands are in original-frame coordinates, so drawing uses scale 1
        draw_scale_x = 1.0
        draw_scale_y = 1.0

        # Draw hands
        for hand in self.latest_hands:
            x, y, w_bbox, h_bbox = hand['bbox']
            x_orig = int(x * draw_scale_x)
            y_orig = int(y * draw_scale_y)
            w_orig = int(w_bbox * draw_scale_x)
            h_orig = int(h_bbox * draw_scale_y)
            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (255, 0, 255), 3)

            lmList = hand['lmList']
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            for p1_idx, p2_idx in connections:
                p1 = lmList[p1_idx]
                p2 = lmList[p2_idx]
                pt1 = (int(p1[0] * draw_scale_x), int(p1[1] * draw_scale_y))
                pt2 = (int(p2[0] * draw_scale_x), int(p2[1] * draw_scale_y))
                cv2.line(frame, pt1, pt2, RED, 2)

            for x_lm, y_lm, _ in lmList:
                center = (int(x_lm * draw_scale_x), int(y_lm * draw_scale_y))
                cv2.circle(frame, center, 5, GREEN, cv2.FILLED)

        self.current_frame_bgr = frame if frame is not None else self.current_frame_bgr

        # Convert to RGB and display
        try:
            img_rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            self.root.after(self.FRAME_DELAY_MS, self.update_frame)
            return

        pil_img = Image.fromarray(img_rgb)
        canvas_w, canvas_h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        src_w, src_h = pil_img.size
        scale = min(canvas_w / src_w, canvas_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil_resized)

        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.coords(self._canvas_img_id, x, y)
        self.canvas.itemconfig(self._canvas_img_id, image=imgtk)
        self.canvas.image = imgtk

        self.root.after(self.FRAME_DELAY_MS, self.update_frame)


    def run(self):
        self.update_frame()
        self.root.mainloop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()