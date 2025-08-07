import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'  # Disable file watcher
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'  # Disable auto-rerun
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'  # Disable usage stats

import cv2
import numpy as np
import time
import logging
import torch
from ultralytics import YOLO
from datetime import datetime
import streamlit as st

# Configure Streamlit
st.set_page_config(
    page_title="Grape Detection System",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "last_30_7.pt"

# Configuration
CONFIG = {
    "TARGET_CLASSES": ["yellow", "black", "stick"],
    "CONFIDENCE": 0.8,
    "CAMERA_SOURCE": 0
}

# Current detection state
STATE = {
    "current_detection_mode": "RED",  # Default mode
}

# Bounding box colors (BGR)
BOX_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "black": (0, 0, 0),
    "stick": (128, 128, 128),
}

CLASS_NAME_OVERRIDE = {"brown": "yellow"}

# Sharpening kernel
SHARPEN_KERNEL = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

# Logging setup
def setup_logging():
    # Create the logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Set the log file path inside the logs directory
    log_file = os.path.join("logs", f"grape_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
def log_event(message):
    logging.info(message)

def update_target_classes(excluded_color):
    """Update target classes to exclude the selected color from detection."""
    global CONFIG, STATE
    all_colors = {
        "RED": "red",
        "BLACK": "black",
        "YELLOW": "yellow",
        "STICK":"stick"
    }
    if excluded_color in all_colors:
        active_classes = [
            color_name for name, color_name in all_colors.items()
            if name != excluded_color
        ]
        CONFIG["TARGET_CLASSES"] = active_classes + ["stick"]
        STATE["current_detection_mode"] = excluded_color
        log_event(f"🎯 Detection will EXCLUDE {excluded_color}, targeting: {', '.join(active_classes + ['stick']).upper()}")

def initialize_yolo_model(model_path=MODEL_PATH):
    """Load YOLO model and return class names."""
    try:
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model = model.to('cuda')
            log_event("🚀 Running YOLO model on GPU")
        else:
            log_event("🖥️ Running YOLO model on CPU")
        log_event("✅ YOLOv8 model loaded successfully from 'last_30_7.pt'")
        class_names = model.names
        log_event(f"📦 Number of classes in model: {len(class_names)}")
        return model, class_names
    except Exception as e:
        logging.error(f"❌ Failed to load YOLO model: {e}")
        st.error(f"Failed to load YOLO model: {e}")
        return None, None

def initialize_video_capture(max_attempts=3, delay=1):
    """Initialize system webcam capture with retries."""
    attempt = 1
    while attempt <= max_attempts:
        try:
            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(CONFIG["CAMERA_SOURCE"])
            if cap.isOpened():
                log_event("📷 System webcam opened successfully")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                log_event(f"🎥 System webcam resolution: {width}x{height}, FPS: {fps}")
                return cap, width, height, fps
            else:
                cap.release()
                log_event(f"❌ System webcam not available at index 0 (Attempt {attempt}/{max_attempts})")
        except Exception as e:
            log_event(f"❌ Error accessing system webcam (Attempt {attempt}/{max_attempts}): {e}")
        
        attempt += 1
        time.sleep(delay)
    
    st.error("❌ System webcam not found after multiple attempts. Please ensure your camera is enabled and not in use.")
    return None, None, None, None

def process_frame(frame, model, width, max_process_time=0.5):
    """Process a single frame with a timeout to prevent freezing."""
    global CONFIG
    try:
        t_start = time.time()
        annotated_frame = frame.copy()
        t_preprocess = time.time()
        log_event(f"frame_copy_time: {t_preprocess - t_start:.3f}s")

        if t_preprocess - t_start > max_process_time:
            log_event("⚠️ Frame copy took too long, skipping processing")
            return frame, [0, 0, 0], False

        results = model(annotated_frame, imgsz=320)  # Reduced for faster inference
        t_inference = time.time()
        log_event(f"model_inference_time: {t_inference - t_preprocess:.3f}s")

        if t_inference - t_preprocess > max_process_time:
            log_event("⚠️ Model inference took too long, skipping processing")
            return frame, [0, 0, 0], False

        zone_width = width // 3
        led_control = [0, 0, 0]
        found_detection = False
        detection_count = 0

        target_classes = [c.lower().strip() for c in CONFIG["TARGET_CLASSES"]]
        for r in results:
            for box in r.boxes:
                if detection_count >= 10:
                    break
                    
                class_id = int(box.cls[0])
                original_class_name = model.names[class_id]
                class_name = CLASS_NAME_OVERRIDE.get(original_class_name, original_class_name).lower().strip()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = BOX_COLORS.get(class_name, (255, 255, 255))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if conf > 0.7:
                    log_event(f"📌 Detected: '{original_class_name}' (as '{class_name}') with confidence {conf:.2f}")

                cx = (x1 + x2) // 2
                line_idx = min(cx // zone_width, 2)
                if class_name in target_classes and conf > CONFIG["CONFIDENCE"]:
                    led_control[line_idx] = 1
                    found_detection = True
                    log_event(f"✅ MATCH: {class_name.upper()} in Line {line_idx + 1}")
                elif conf > 0.6:
                    log_event(f"🚫 IGNORED: '{class_name}' accept targets or low confidence")
                
                detection_count += 1

        t_end = time.time()
        log_event(f"bounding_box_processing_time: {t_end - t_inference:.3f}s")
        log_event(f"total_process_frame_time: {t_end - t_start:.3f}s")
        return annotated_frame, led_control, found_detection
    except Exception as e:
        logging.error(f"❌ Frame processing error: {e}")
        st.error(f"Frame processing error: {e}")
        return frame, [0, 0, 0], False

def draw_zone_lines(frame, width, height, led_control):
    """Draw full rectangular borders around each zone with status-based coloring."""
    try:
        zone_width = width // 3
        default_color = (255, 255, 0)  # Yellow (inactive)
        active_color = (0, 0, 255)     # Red (active)
        font_scale = 0.6
        font_thickness = 2

        for i in range(3):
            x_start = i * zone_width
            x_end = (i + 1) * zone_width if i < 2 else width
            zone_color = active_color if led_control[i] == 1 else default_color

            # Draw full rectangle (4 sides) as border for the zone
            cv2.rectangle(frame, (x_start, 0), (x_end, height), zone_color, thickness=3)

            # Draw zone label inside each rectangle
            label_x = x_start + 10
            label_y = 30
            cv2.putText(frame, f"Line {i+1}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, zone_color, font_thickness)

        # Show detection mode at bottom
        cv2.putText(frame, f"Mode: {STATE['current_detection_mode']}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        logging.error(f"❌ Zone lines drawing error: {e}")
def cleanup_resources(cap=None):
    """Clean up camera resources."""
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            log_event("📷 Webcam released successfully")
        log_event("🧹 Resources cleaned up successfully")
    except Exception as e:
        logging.error(f"❌ Error during resource cleanup: {e}")
        st.error(f"Error during resource cleanup: {e}")

def main():
    """Main function to run the grape detection system with Streamlit UI."""
    try:
        st.title("Raisins Color Sorting System")
        
        # Initialize session state
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        if 'model_initialized' not in st.session_state:
            st.session_state.model_initialized = False
        if 'led_status' not in st.session_state:
            st.session_state.led_status = []
        if 'found_detection' not in st.session_state:
            st.session_state.found_detection = False
        if 'last_frame_time' not in st.session_state:
            st.session_state.last_frame_time = time.time()
        if 'frame_rate' not in st.session_state:
            st.session_state.frame_rate = 25

        with st.sidebar:
            st.header("🎛️ Detection Controls")
            color_option = st.selectbox("Sorting Color", ["RED", "BLACK", "YELLOW"], index=0, key="color_selector")
            
            if color_option != STATE["current_detection_mode"]:
                update_target_classes(color_option)
                st.success(f"Detection mode updated to exclude: {color_option}")
            
            st.header("⚙️ Performance Settings")
            new_frame_rate = st.slider("Frame Rate (FPS)", min_value=1, max_value=10, value=st.session_state.frame_rate, key="frame_rate")
            
            if new_frame_rate != st.session_state.frame_rate:
                st.session_state.frame_rate = new_frame_rate
                log_event(f"📊 Frame rate updated to {st.session_state.frame_rate} FPS")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Detection", key="start_btn"):
                    st.session_state.detection_active = True
                    st.session_state.model_initialized = False
            with col2:
                if st.button("⏹️ Stop Detection", key="stop_btn"):
                    st.session_state.detection_active = False
                    cleanup_resources(st.session_state.get('cap'))
                    st.session_state.model_initialized = False
                    st.success("Detection stopped!")
            
            st.header("📊 Status")
            if st.session_state.detection_active:
                st.success("🟢 Detection Active")
            else:
                st.warning("🟡 Detection Stopped")
            
            if st.session_state.led_status:
                st.write("💡 LED Control Status:")
                for status in st.session_state.led_status:
                    st.write(status)
            if st.session_state.found_detection:
                st.success("🎯 Detection found!")
            else:
                st.info("🔍 No detection in current frame")

        if st.session_state.detection_active:
            if not st.session_state.model_initialized:
                with st.spinner("🔧 Initializing detection system..."):
                    setup_logging()
                    os.makedirs("missed_frames", exist_ok=True)
                    model, _ = initialize_yolo_model()
                    
                    if model is None:
                        st.error("❌ Failed to initialize model.")
                        st.session_state.detection_active = False
                        return
                    
                    cap, width, height, fps = initialize_video_capture()
                    if cap is None:
                        st.error("❌ Failed to initialize camera.")
                        st.session_state.detection_active = False
                        return
                    
                    st.session_state.model = model
                    st.session_state.cap = cap
                    st.session_state.width = width
                    st.session_state.height = height
                    st.session_state.model_initialized = True
                    log_event(f"🚀 Starting detection in {STATE['current_detection_mode']} mode")
                    st.success("✅ Detection system initialized!")
            
            placeholder = st.empty()
            frame_interval = 1.0 / st.session_state.frame_rate
            last_display_time = time.time()
            display_interval = 0.2  # Update UI every 0.2 seconds

            while st.session_state.detection_active:
                current_time = time.time()
                if current_time - st.session_state.last_frame_time >= frame_interval:
                    ret, frame = st.session_state.cap.read()
                    if ret:
                        annotated_frame, led_control, found_detection = process_frame(frame, st.session_state.model, st.session_state.width)
                        draw_zone_lines(annotated_frame, st.session_state.width, st.session_state.height, led_control)

                        if current_time - last_display_time >= display_interval:
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            placeholder.image(frame_rgb, caption=f"📹 Live Feed ({st.session_state.frame_rate} FPS)", use_container_width=True)
                            last_display_time = current_time

                        st.session_state.led_status = [f"Line {i+1}: {'🟢 ON' if led == 1 else '🔴 OFF'}" for i, led in enumerate(led_control)]
                        st.session_state.found_detection = found_detection
                        st.session_state.last_frame_time = current_time
                    else:
                        st.error("❌ Failed to read frame from camera")
                        st.session_state.detection_active = False
                        cleanup_resources(st.session_state.get('cap'))
                        break
                
                time.sleep(0.01)  # Prevent CPU overload
        else:
            st.info("⏸️ Detection stopped. Click 'Start Detection' to begin.")
            cleanup_resources(st.session_state.get('cap'))
    finally:
        cleanup_resources(st.session_state.get('cap'))
        st.session_state.model_initialized = False

if __name__ == "__main__":
    main()