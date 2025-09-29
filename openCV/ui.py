import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import sys

class HandGUI:
    def __init__(self, model=None, class_names=None):
        self.model = model
        self.class_names = class_names or []
        self.img_height, self.img_width = 64, 64

        # Prediction buffer (like typed text)
        self.global_variable = ""

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
        self.variable_label = tk.Label(
            self.bottom_frame,
            text="Waiting for detection...",
            font=("TkDefaultFont", 12),
            anchor="w", justify="left"
        )
        self.variable_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bottom_frame.configure(height=42)

        self.copy_btn = tk.Button(self.bottom_frame, text="Copy", command=self.copy_variables)
        self.copy_btn.pack(side=tk.RIGHT, padx=(8, 0))

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

    def copy_variables(self):
        text = self.variable_label.cget("text")
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.copy_btn.config(text="Copied!")
            self.root.after(1000, lambda: self.copy_btn.config(text="Copy"))
        except Exception:
            self.copy_btn.config(text="Error")
            self.root.after(1000, lambda: self.copy_btn.config(text="Copy"))

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def predict_hand(self, frame, hand_box):
        if self.model is None:
            return None

        x, y, w, h = hand_box["bbox"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        hand_img = frame[y1:y2, x1:x2]

        if hand_img.size == 0:
            return None

        hand_img = cv2.resize(hand_img, (self.img_width, self.img_height))
        hand_img = hand_img.astype("float32") / 127.5 - 1.0
        hand_img = np.expand_dims(hand_img, axis=0)

        preds = self.model.predict(hand_img, verbose=0)
        class_id = int(np.argmax(preds[0]))
        print(class_id, self.class_names[class_id], preds[0][class_id])
        return self.class_names[class_id], float(preds[0][class_id])

    def apply_prediction(self, pred_class):
        """Update global_variable text based on prediction class."""
        if pred_class == "del":
            self.global_variable = self.global_variable[:-1]
        elif pred_class == "space":
            self.global_variable += " "
        elif pred_class == "nothing":
            pass  # ignore
        else:
            self.global_variable += pred_class

        self.variable_label.config(text=self.global_variable)

    def update_frame(self):
        self._frame_counter += 1
        success, frame = self.cap.read()
        if not success or frame is None or frame.size == 0:
            self.root.after(self.FRAME_DELAY_MS, self.update_frame)
            return
        
        GREEN = (0, 255, 0) 
        RED = (0, 0, 255)  
        frame = cv2.flip(frame, 1)
        try:
            h, w = frame.shape[:2]
            scale = self.DETECT_WIDTH / float(w) if w > self.DETECT_WIDTH else 1.0
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        except Exception:
            small = frame

        if (self._frame_counter % self.PROCESS_EVERY_N) == 0:
            try:
                hands, _ = self.detector.findHands(small, draw=False) 
                self.latest_hands = hands if hands is not None else []
            except Exception:
                self.latest_hands = []

            if self.latest_hands:
                result = self.predict_hand(frame, self.latest_hands[0])
                if result:
                    pred_class, conf = result
                    self.apply_prediction(pred_class)
    
        draw_scale_x = w / small.shape[1]
        draw_scale_y = h / small.shape[0]

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

            for x, y, _ in lmList:
                center = (int(x * draw_scale_x), int(y * draw_scale_y))
                cv2.circle(frame, center, 5, GREEN, cv2.FILLED)
        self.current_frame_bgr = frame

        if self.current_frame_bgr is None:
            self.current_frame_bgr = frame

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