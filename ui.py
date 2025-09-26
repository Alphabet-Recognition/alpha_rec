import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk
import sys

# Performance tuning: detection on a smaller image and skip frames
DETECT_WIDTH = 640           # width to resize camera frame for detection (smaller -> faster)
PROCESS_EVERY_N = 2           # run heavy detector every N frames (>=1)
FRAME_DELAY_MS = 15           # GUI loop delay in ms (was 10)

# Initialize Hand Detector with lower detection confidence for speed
detector = HandDetector(maxHands=1, detectionCon=0.5)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera", file=sys.stderr)
    sys.exit(1)

camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# Tkinter setup
root = tk.Tk()
root.title("Hand Tracking and Variable Display")

# Canvas for camera feed
canvas = tk.Canvas(root, width=camera_width, height=camera_height)
canvas.pack(fill=tk.BOTH, expand=True)

# keep reference to canvas image item
_canvas_img_id = canvas.create_image(0, 0, anchor=tk.NW)
canvas.image = None

# Create a bottom overlay frame placed on the canvas so it stays visible even when the window is small
bottom_frame = tk.Frame(canvas, bd=1, relief=tk.FLAT)
# create a window on the canvas that will hold the bottom_frame (anchor south -> bottom-center placement)
_bottom_window_id = canvas.create_window(0, 0, anchor='s', window=bottom_frame)
# Label for variable display (left side of the bottom_frame)
variable_label = tk.Label(bottom_frame, text="Variables will be displayed here",
                          font=("TkDefaultFont", 12), anchor="w", justify="left")
variable_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6,6), pady=6)
# ensure the overlay has a minimum height so it remains readable at very small window sizes
bottom_frame.configure(height=42)

def _reposition_bottom_overlay(event=None):
    """Keep the bottom overlay anchored bottom-center and adapt wraplength."""
    cw = max(1, canvas.winfo_width())
    ch = max(1, canvas.winfo_height())
    # place at bottom center with a small margin
    margin_y = 12
    canvas.coords(_bottom_window_id, cw // 2, ch - margin_y)
    # set bottom_frame width so it fits inside canvas with margins
    new_w = max(60, cw - 24)
    bottom_frame.configure(width=new_w)
    # give the label a wraplength so text will wrap instead of being clipped
    # leave space for the button (approx 80 px)
    variable_label.configure(wraplength=max(20, new_w - 80))

# bind canvas resize to reposition overlay
canvas.bind("<Configure>", _reposition_bottom_overlay)
# initial reposition after widgets are realized
root.after(50, _reposition_bottom_overlay)

def copy_variables():
    """Copy current variable text to the clipboard and show a brief confirmation."""
    text = variable_label.cget("text")
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        copy_btn.config(text="Copied!")
        root.after(1000, lambda: copy_btn.config(text="Copy"))
    except Exception:
        copy_btn.config(text="Error")
        root.after(1000, lambda: copy_btn.config(text="Copy"))

copy_btn = tk.Button(bottom_frame, text="Copy", command=copy_variables)
copy_btn.pack(side=tk.RIGHT, padx=(8, 0))

# Global variables (example)
global_variable = "asdsadsadasdasdasdsasdasdasdsadasasdasdasdsadsadsadasdssdaasasdasdsdaasd"

# store latest BGR frame
current_frame_bgr = None
# store latest HandDetector solution (list of hand dicts / structures)
latest_hands = []  # will be updated each frame with detector output (or cleared on failure)
# simple frame counter to skip processing on some frames
_frame_counter = 0

def update_variables():
    """Updates the variable display label."""
    global global_variable1, global_variable2
    variable_label.config(text=f"Variable 1: {global_variable}")
    root.after(200, update_variables)  # Update every 200 ms

def on_close():
    """Cleanup on window close."""
    if cap.isOpened():
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

def update_frame():
    """Capture camera, run detector, and draw to canvas scaled to window."""
    global current_frame_bgr, _canvas_img_id, latest_hands, _frame_counter

    # increment frame counter for skipping logic
    _frame_counter += 1

    success, frame = cap.read()
    if not success or frame is None or frame.size == 0:
        # try again later
        root.after(FRAME_DELAY_MS, update_frame)
        return

    # keep BGR frame for any processing/resizing
    frame = cv2.flip(frame, 1)
    # Resize for faster detection
    try:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / float(w) if w > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    except Exception:
        small = frame

    # Run detector only every PROCESS_EVERY_N frames on the smaller image
    if (_frame_counter % PROCESS_EVERY_N) == 0:
        try:
            hands, processed_small = detector.findHands(small, draw=True)
            latest_hands = hands if hands is not None else []
            # upscale processed_small to original capture size for display quality
            try:
                processed = cv2.resize(processed_small, (w, h), interpolation=cv2.INTER_LINEAR)
                current_frame_bgr = processed
            except Exception:
                current_frame_bgr = frame
        except Exception:
            # if detector fails, fallback to raw frame
            current_frame_bgr = frame
            latest_hands = []
    else:
        # reuse last processed frame (faster) or raw frame if none yet
        if current_frame_bgr is None:
            current_frame_bgr = frame

    # convert to PIL (RGB)
    try:
        img_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        root.after(FRAME_DELAY_MS, update_frame)
        return

    pil_img = Image.fromarray(img_rgb)

    # scale to canvas current size while preserving aspect (fit)
    canvas_w = max(1, canvas.winfo_width())
    canvas_h = max(1, canvas.winfo_height())

    # compute aspect-fit size
    src_w, src_h = pil_img.size
    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    imgtk = ImageTk.PhotoImage(pil_resized)

    # center image on canvas
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.coords(_canvas_img_id, x, y)
    canvas.itemconfig(_canvas_img_id, image=imgtk)
    canvas.image = imgtk  # keep reference

    root.after(FRAME_DELAY_MS, update_frame)

# Start loops
update_variables()
update_frame()

root.mainloop()

# Cleanup (in case)
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()