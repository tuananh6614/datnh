import os
import re
import cv2
import sys
import json

import time
import socket
import random
import string
import datetime
import subprocess
import threading
import collections
import logging
import requests
import psycopg2
import psycopg2.extras
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging - CHỈ log ra console, không ghi file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # ✅ Chỉ log ra console
    ]
)

# Torch để tự phát hiện GPU
try:
    import torch
except Exception:
    torch = None

# YOLOv8
from ultralytics import YOLO

# EasyOCR
import easyocr

# MQTT (tùy chọn, nếu không cài sẽ tự OFF)
try:
    from paho.mqtt import client as mqtt
except Exception:
    mqtt = None

# PySide6
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLineEdit, QPushButton, QStatusBar, QMessageBox,
    QSizePolicy, QDialog, QComboBox, QDialogButtonBox, QFormLayout, QCheckBox,
    QScrollArea  # Responsive layout
)

# ✅ QR Payment Display Widget
from qr_payment_widget import QRPaymentWidget
from sepay_helper import create_parking_payment_sepay, SEPAY_CONFIG, sepay

# =================================================================================================
# CẤU HÌNH
# =================================================================================================

CFG_FILE = "config.json"

# Đường dẫn model YOLO
YOLO_MODEL_PATH = r"D:\FIRMWAVE\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt"

# Thư mục lưu ảnh
DIR_IN  = Path("plates/IN")
DIR_OUT = Path("plates/OUT")
DIR_IN.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Tham số ALPR
YOLO_CONF     = 0.35
YOLO_IMGSZ    = 416
MIN_REL_AREA  = 0.010
MIN_SHARPNESS = 60.0
CAP_WIDTH, CAP_HEIGHT = 640, 480

# Multi-Frame Voting
# ✅ TỐI ƯU: Tách riêng IN và OUT để giảm lag
VOTE_FRAMES_IN  = 4  # IN cần độ chính xác cao (biển số mới) - giảm từ 7
VOTE_FRAMES_OUT = 3  # OUT chỉ cần khớp với biển số đã có - giảm từ 7
VOTE_FRAMES     = 7  # Legacy - deprecated
VOTE_GAP_MS     = 30
VOTE_MIN_HITS   = 2

# Perspective warp (nắn hình)
WARP_W, WARP_H = 320, 96

# Phí gửi xe
FEE_BASE = 3000       # Phí vào cổng
FEE_INCREMENT = 5000  # Tăng mỗi khoảng thời gian
FEE_INTERVAL = 60      # Khoảng thời gian (phút)

def calculate_parking_fee(duration_minutes):
    """
    Tính phí gửi xe: Vào 3,000đ, mỗi 60 phút (1 giờ) tăng 5,000đ

    Args:
        duration_minutes (int): Thời gian gửi xe (phút)

    Returns:
        int: Phí gửi xe (VNĐ)
    """
    blocks = int(duration_minutes / FEE_INTERVAL)
    total_fee = FEE_BASE + (blocks * FEE_INCREMENT)
    return total_fee

# Regex biển VN (rộng)
PLATE_RE = re.compile(r"[0-9]{2,3}[A-Z]{1,2}[-\s]?[0-9]{3,5}")

# =================================================================================================
# EXCEPTION HANDLER
# =================================================================================================

def exception_hook(exctype, value, traceback):
    logging.error("Uncaught exception", exc_info=(exctype, value, traceback))
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = exception_hook

# =================================================================================================
# DATA CLASS CẤU HÌNH UI
# =================================================================================================

@dataclass
class UiConfig:
    cam_in_index: int = 0
    cam_out_index: int = -1
    total_slots: int = 50
    mqtt_enable: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    gate_id: str = "gate01"
    auto_start_broker: bool = False
    broker_exe: str = r"C:\Program Files\mosquitto\mosquitto.exe"
    broker_conf: str = r"D:\FIRMWAVE\project\mosquitto.conf"
    # Server & Database config (cho remote access)
    server_url: str = "https://parking.epulearn.xyz"
    db_host: str = "192.168.1.165"
    db_port: int = 5432
    db_name: str = "parking_system"
    db_user: str = "parking_admin"
    db_password: str = "parking123"

def load_config() -> UiConfig:
    if os.path.exists(CFG_FILE):
        try:
            with open(CFG_FILE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            defaults = UiConfig().__dict__
            data = {k: d.get(k, defaults[k]) for k in defaults.keys()}
            return UiConfig(**data)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
    return UiConfig()

def save_config(cfg: UiConfig):
    with open(CFG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, ensure_ascii=False, indent=2)

# =================================================================================================
# CÔNG CỤ ẢNH / VIDEO
# =================================================================================================

def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_for_plate(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def np_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_copy = rgb.copy()  # ✅ FIX: Copy data
    h, w, ch = rgb_copy.shape
    return QImage(rgb_copy.data, w, h, ch*w, QImage.Format_RGB888).copy()

def set_pixmap_fit_no_upscale(label: QLabel, img: QImage):
    try:
        if label.width() <= 0 or label.height() <= 0 or img.isNull():
            return
        pix = QPixmap.fromImage(img)
        sw, sh = label.width() / pix.width(), label.height() / pix.height()
        scale = min(1.0, sw, sh)
        new_size = QSize(int(pix.width()*scale), int(pix.height()*scale))
        label.setPixmap(pix.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)
    except Exception as e:
        logging.error(f"Error in set_pixmap_fit_no_upscale: {e}")

def list_cameras(max_index=8) -> List[int]:
    found = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
                cap.release()
        except Exception as e:
            logging.warning(f"Error checking camera {i}: {e}")
    return found

# =================================================================================================
# CÔNG CỤ XỬ LÝ BIỂN SỐ
# =================================================================================================

def plate_similarity(plate1: str, plate2: str) -> float:
    """Tính độ tương đồng giữa 2 biển số (0.0 - 1.0)"""
    if not plate1 or not plate2:
        return 0.0

    # Chuẩn hóa: bỏ khoảng trắng, dấu gạch ngang, chữ hoa
    p1 = plate1.upper().replace(" ", "").replace("-", "")
    p2 = plate2.upper().replace(" ", "").replace("-", "")

    if p1 == p2:
        return 1.0

    # Tính Levenshtein distance (edit distance)
    len1, len2 = len(p1), len(p2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Dynamic programming matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if p1[i-1] == p2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[len1][len2]
    max_len = max(len1, len2)
    similarity = 1.0 - (edit_distance / max_len)

    return similarity

# =================================================================================================
# CÔNG CỤ FILE
# =================================================================================================

def cleanup_old_images(days_old=3):
    """Xóa ảnh cũ hơn N ngày"""
    try:
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for root_dir in [DIR_IN, DIR_OUT]:
            if not root_dir.exists():
                continue

            # Duyệt qua tất cả file trong thư mục và thư mục con
            for file_path in root_dir.rglob("*.jpg"):
                try:
                    # Kiểm tra thời gian modified
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()  # Xóa file
                        deleted_count += 1
                        logging.debug(f"Deleted old image: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {e}")

        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} images older than {days_old} days")
        else:
            logging.info(f"No images older than {days_old} days found")

    except Exception as e:
        logging.error(f"Error in cleanup_old_images: {e}")

# =================================================================================================
# CÔNG CỤ MẠNG / MQTT
# =================================================================================================

def is_port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def get_local_ips() -> set:
    ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips

# =================================================================================================
# CAMERA THREAD (✅ FIXED)
# =================================================================================================

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    opened = Signal(bool)
    error_occurred = Signal(str)  # ✅ NEW: Signal cho lỗi

    def __init__(self, source=0, width=CAP_WIDTH, height=CAP_HEIGHT, mirror=False, parent=None):
        super().__init__(parent)
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        self._running = False
        self._buffer = collections.deque(maxlen=25)
        self._buf_lock = threading.Lock()
        self.cap = None

    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        consecutive_errors = 0
        max_errors = 10
        
        try:
            self.cap = cv2.VideoCapture(self.source, backend)
            if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok = self.cap.isOpened()
            self.opened.emit(ok)
            if not ok:
                logging.error(f"Camera {self.source} failed to open")
                return

            target_dt = 1/25.0
            last_emit = 0.0
            
            while self._running:
                if not self._running:  # ✅ Double check
                    break
                    
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            logging.error(f"Camera {self.source}: Too many errors")
                            self.error_occurred.emit("Camera lỗi liên tục!")
                            break
                        QThread.msleep(50)
                        continue
                    
                    consecutive_errors = 0  # ✅ Reset on success
                    
                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    # Calculate sharpness
                    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    score = sharpness_score(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                    
                    with self._buf_lock:
                        self._buffer.append((score, frame.copy()))

                    # Emit frame for display
                    if time.time() - last_emit >= target_dt:
                        disp = frame
                        h0, w0 = frame.shape[:2]
                        if w0 > 640:
                            scale = 640 / w0
                            disp = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
                        
                        # ✅ FIX: Proper data copy
                        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        rgb_copy = rgb.copy()
                        h, w, ch = rgb_copy.shape
                        qimg = QImage(rgb_copy.data, w, h, ch*w, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qimg)
                        last_emit = time.time()

                    rem = target_dt - (time.time() - time.time())
                    if rem > 0:
                        QThread.msleep(int(rem*1000))
                        
                except Exception as e:
                    logging.error(f"Error in camera loop: {e}")
                    consecutive_errors += 1
                    
        except Exception as e:
            logging.error(f"Camera thread crashed: {e}", exc_info=True)
        finally:
            self._running = False
            QThread.msleep(100)  # ✅ Wait for any pending operations
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    logging.info(f"Camera {self.source} released")
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")

    def stop(self):
        """✅ FIXED: Proper shutdown sequence"""
        logging.info(f"Stopping camera {self.source}...")
        self._running = False
        
        # Wait for thread to finish FIRST
        if not self.wait(3000):  # ✅ Increased timeout
            logging.warning(f"Camera {self.source} thread didn't stop gracefully")
            self.terminate()
            self.wait(1000)
        
        # Then release camera
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logging.info(f"Camera {self.source} released after stop")
        except Exception as e:
            logging.error(f"Error releasing camera in stop(): {e}")

    def get_recent_frames(self, n: int, min_score: float = MIN_SHARPNESS, gap_ms: int = 0) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        with self._buf_lock:
            if not self._buffer:
                return frames
            sorted_buf = sorted(list(self._buffer), key=lambda t: -t[0])
        for s, f in sorted_buf:
            if s < min_score:
                continue
            frames.append(f.copy())
            if len(frames) >= n:
                break
        if gap_ms > 0 and len(frames) >= 2:
            frames = frames[::2] if len(frames) > n else frames
            frames = frames[:n]
        return frames

    def best_recent_frame(self, min_score: float = MIN_SHARPNESS) -> Optional[np.ndarray]:
        with self._buf_lock:
            if not self._buffer:
                return None
            s, f = max(self._buffer, key=lambda t: t[0])
            return f.copy() if s >= min_score else None

# =================================================================================================
# ALPR (✅ FIXED: Cache management + cleanup)
# =================================================================================================

def clean_plate_text(txt: str) -> str:
    t = txt.upper().replace("O", "0")
    t = re.sub(r"[^A-Z0-9\s-]", "", t)
    m = PLATE_RE.search(t.replace(" ", ""))
    if not m:
        return t.strip()
    raw = m.group(0)
    if "-" not in raw:
        if len(raw) > 3 and raw[2].isalpha():
            raw = raw[:2] + "-" + raw[2:]
        elif len(raw) > 4 and raw[3].isalpha():
            raw = raw[:3] + "-" + raw[3:]
    raw = re.sub(r"-([A-Z]{1,2})(\d+)", r"-\1 \2", raw)
    return raw.strip()

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_plate(crop_bgr: np.ndarray, out_w=WARP_W, out_h=WARP_H) -> np.ndarray:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(crop_bgr, (out_w, out_h))

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * (crop_bgr.shape[0]*crop_bgr.shape[1]):
        return cv2.resize(crop_bgr, (out_w, out_h))

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)

    src = order_points(box.astype(float)).astype(np.float32)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    if src.shape != (4,2):
        return cv2.resize(crop_bgr, (out_w, out_h))
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop_bgr, M, (out_w, out_h))
    return warped

class ALPR:
    def __init__(self, weights: str, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, 
                 max_workers=4, cache_ttl=5.0, max_cache_size=100):
        # Kiểm tra GPU
        self.device = 'cuda' if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else 'cpu'
        logging.info(f"ALPR using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
            logging.info(f"YOLO model loaded on {self.device}")
        except Exception as e:
            logging.warning(f"Failed to move YOLO to GPU: {e}")
            self.device = 'cpu'
            
        self.conf = conf
        self.imgsz = imgsz

        logging.info(f"Initializing EasyOCR with GPU=False (RTX 5050 compatibility)")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logging.info(f"EasyOCR initialized successfully")
        
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # ✅ FIX: Cache with size limit
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: str) -> Optional[str]:
        with self._cache_lock:
            now = time.time()
            if key in self.cache:
                txt, ts = self.cache[key]
                if now - ts < self.cache_ttl:
                    return txt
                else:
                    try:
                        del self.cache[key]
                    except Exception:
                        pass
            return None

    def _cache_put(self, key: str, text: str):
        with self._cache_lock:
            now = time.time()
            
            # ✅ FIX: Cleanup old cache if too large
            if len(self.cache) >= self.max_cache_size:
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                to_remove = len(sorted_items) // 5  # Remove 20%
                for k, _ in sorted_items[:to_remove]:
                    try:
                        del self.cache[k]
                    except Exception:
                        pass
                logging.info(f"Cache cleanup: removed {to_remove} entries")
            
            self.cache[key] = (text, now)

    def cleanup(self):
        """✅ NEW: Cleanup when closing app"""
        logging.info("Cleaning up ALPR resources...")
        try:
            self.pool.shutdown(wait=True, timeout=5)
        except Exception as e:
            logging.error(f"Error shutting down thread pool: {e}")
        
        with self._cache_lock:
            self.cache.clear()
        
        logging.info("ALPR cleanup completed")

    def infer_once(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        debug = frame.copy()
        H, W = frame.shape[:2]
        try:
            results = self.model(
                frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
        except TypeError:
            results = self.model(
                frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
        except Exception:
            results = self.model(
                frame, device='cpu', conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
            
        if results.boxes is None or len(results.boxes) == 0:
            return None, debug
            
        best_txt, best_score = None, -1.0
        confs = results.boxes.conf.detach().cpu().numpy()
        order = np.argsort(-confs)
        
        for idx in order:
            b = results.boxes[int(idx)]
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W-1,x2), min(H-1,y2)
            w,h = x2-x1, y2-y1
            if w <= 1 or h <= 1: continue
            
            rel_area = (w*h)/(W*H)
            if rel_area < MIN_REL_AREA: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            if sharpness_score(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_SHARPNESS: continue
            
            key = f"{x1}-{y1}-{x2}-{y2}"
            cached = self._cache_get(key)
            
            if cached:
                text = cached
            else:
                warped = warp_plate(crop, WARP_W, WARP_H)
                warped = enhance_for_plate(warped)
                dets = self.reader.readtext(warped)
                text = " ".join([d[1] for d in dets]) if dets else ""
                text = clean_plate_text(text)
                if text: self._cache_put(key, text)
            
            score = float(b.conf.item()) + 0.05*len(text)
            if text and score > best_score:
                best_score = score
                best_txt = text
                
            cv2.rectangle(debug, (x1,y1), (x2,y2), (0,255,0), 2)
            dbg_txt = text if text else "?"
            cv2.putText(debug, dbg_txt, (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                        
        return best_txt, debug

    def infer_multi(self, frames: List[np.ndarray]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        if not frames:
            return None, None
            
        futures = {self.pool.submit(self.infer_once, f): f for f in frames}
        votes: Dict[str, int] = {}
        best_debug = None
        best_plate = None
        best_score = -1
        
        for fut in as_completed(futures):
            try:
                plate, debug = fut.result()
                if plate:
                    votes[plate] = votes.get(plate, 0) + 1
                    cur_score = votes[plate]*10 + len(plate)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_plate = plate
                        best_debug = debug
            except Exception as e:
                logging.error(f"Error in infer_multi: {e}")
                
        if not votes:
            return None, frames[0].copy()
            
        max_hits = max(votes.values())
        cands = [p for p, c in votes.items() if c == max_hits]
        cands.sort(key=lambda s: (-len(s), s))
        plate_final = cands[0]
        
        if max_hits < VOTE_MIN_HITS:
            return None, best_debug
            
        return plate_final, best_debug

# =================================================================================================
# UI PHỤ TRỢ
# =================================================================================================

def qlabel_video_placeholder(text=""):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setMinimumSize(QSize(200, 150))  # Giảm minimum size để responsive tốt hơn
    lbl.setStyleSheet("background:#1f1f1f;color:#cccccc;border:1px solid #3a3a3a;")
    return lbl

# =================================================================================================
# HỘP THOẠI THIẾT LẬP
# =================================================================================================

class SettingsDialog(QDialog):
    def __init__(self, cfg: UiConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt")
        self.resize(420, 320)

        cams = list_cameras()
        self.cb_in  = QComboBox()
        self.cb_out = QComboBox(); self.cb_out.addItem("— Tắt —", -1)
        if not cams:
            self.cb_in.addItem("Không tìm thấy camera", -1)
        else:
            for i in cams:
                self.cb_in.addItem(f"Camera {i}", i)
                self.cb_out.addItem(f"Camera {i}", i)

        if cams and cfg.cam_in_index in cams:
            self.cb_in.setCurrentIndex(cams.index(cfg.cam_in_index))
        if cfg.cam_out_index == -1:
            self.cb_out.setCurrentIndex(0)
        elif cfg.cam_out_index in cams:
            self.cb_out.setCurrentIndex(1 + cams.index(cfg.cam_out_index))

        self.ed_slots  = QLineEdit(str(cfg.total_slots))

        self.chk_mqtt  = QCheckBox("Bật MQTT"); self.chk_mqtt.setChecked(cfg.mqtt_enable)
        self.ed_host   = QLineEdit(cfg.mqtt_host)
        self.ed_port   = QLineEdit(str(cfg.mqtt_port))
        self.ed_gate   = QLineEdit(cfg.gate_id)

        form = QFormLayout()
        form.addRow("Ngõ vào:", self.cb_in)
        form.addRow("Ngõ ra:", self.cb_out)
        form.addRow("SLOT TỔNG:", self.ed_slots)
        form.addRow(self.chk_mqtt)
        form.addRow("MQTT Host:", self.ed_host)
        form.addRow("MQTT Port:", self.ed_port)
        form.addRow("Gate ID:", self.ed_gate)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)

    def values(self):
        return (
            self.cb_in.currentData(), self.cb_out.currentData(), int(self.ed_slots.text() or "1"),
            self.chk_mqtt.isChecked(), self.ed_host.text().strip() or "127.0.0.1",
            int(self.ed_port.text() or "1883"), self.ed_gate.text().strip() or "gate1"
        )

# =================================================================================================
# CỬA SỔ CHÍNH (✅ FIXED: Thread safety + proper cleanup)
# =================================================================================================

class MainWindow(QMainWindow):
    # Qt Signals for thread-safe triggering
    trigger_shoot_in = Signal(str)  # Signal with card_id
    trigger_shoot_out = Signal(str)  # Signal with card_id
    trigger_update_revenue = Signal(str, str, int, int)  # ✅ NEW: Signal(card_id, plate, fee, total_revenue)
    trigger_invalid_card = Signal(str)  # ✅ NEW: Signal for invalid card warning
    trigger_display_qr = Signal(dict)  # ✅ NEW: Signal for QR payment display - dict with payment data
    trigger_payment_auto_confirmed = Signal(int, dict)  # ✅ NEW: Auto-confirm payment signal

    def __init__(self, cfg: UiConfig):
        super().__init__()
        self.cfg = cfg
        self.current_user = None  # Will be set after login
        self.allow_close = False  # ✅ Chặn nút X, chỉ cho phép đóng qua Logout
        self.login_time = None  # ✅ Thời gian đăng nhập

        # Lưu theo mã thẻ: {card_id: {"plate": "...", "time": datetime, "card_id": ..., "paid": False}}
        self._in_records: Dict[str, Dict] = {}
        self._paid_cards: Dict[str, datetime.datetime] = {}  # ✅ NEW: {card_id: paid_time}
        self._rec_lock = threading.RLock()

        # ✅ NEW: Revenue tracking (từ in_records.json)
        self._total_revenue = 0
        self._total_in_count = 0

        self._local_ips = get_local_ips()
        self._mqtt_connected = False
        self._esp_devices: Dict[str, Dict] = {}  # Lưu thông tin nhiều ESP32: {mac: {ip, last_hb, online}}
        self._hb_timeout = 5.0  # Giảm timeout xuống 5 giây
        self._mosq_proc = None
        self.mqtt_client = None
        self._pending_card_id = ""  # Lưu mã thẻ RFID tạm thời
        self.payment_polling_timer: Optional[QTimer] = None  # ✅ NEW: Timer kiểm tra trạng thái thanh toán
        self.payment_polling_session_id: Optional[int] = None
        self.payment_polling_amount: int = 0  # ✅ NEW: Số tiền cần verify
        self.payment_polling_interval = 1000  # ms
        self.payment_polling_count = 0  # ✅ Counter để giới hạn tần suất gọi SePay API

        # Cleanup ảnh cũ (background thread)
        threading.Thread(target=cleanup_old_images, args=(3,), daemon=True).start()

        # Connect signals to slots
        self.trigger_shoot_in.connect(self._handle_shoot_in)
        self.trigger_shoot_out.connect(self._handle_shoot_out)
        self.trigger_update_revenue.connect(self._handle_update_revenue)  # ✅ NEW
        self.trigger_invalid_card.connect(self._handle_invalid_card)  # ✅ NEW
        self.trigger_display_qr.connect(self._handle_display_qr)  # ✅ NEW: Thread-safe QR display
        self.trigger_payment_auto_confirmed.connect(self._on_payment_auto_confirmed)  # ✅ NEW

        self._build_ui()

        # ✅ NEW: Load dữ liệu từ in_records.json SAU KHI build UI
        self._load_data_from_db()

        self._init_models()

        self.cam_in_worker: Optional[CameraWorker] = None
        self.cam_out_worker: Optional[CameraWorker] = None
        self.start_cameras()

        self.ensure_broker_running()
        self.init_mqtt()

        self._start_timers()

    def _get_db_connection(self):
        """Kết nối PostgreSQL database với timezone VN (dùng config)"""
        try:
            conn = psycopg2.connect(
                host=self.cfg.db_host,
                port=self.cfg.db_port,
                database=self.cfg.db_name,
                user=self.cfg.db_user,
                password=self.cfg.db_password,
                options="-c timezone=Asia/Ho_Chi_Minh"  # ✅ Set timezone khi connect
            )
            return conn
        except Exception as e:
            logging.error(f"[DATABASE] Connection error to {self.cfg.db_host}: {e}")
            return None

    def _fetch_latest_unpaid_session(self, card_id: str) -> Optional[dict]:
        """Lấy session chưa thanh toán mới nhất từ database giống logic Flask."""
        if not card_id:
            return None

        conn = self._get_db_connection()
        if not conn:
            return None

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT
                    session_id,
                    plate_number,
                    time_in,
                    time_out,
                    payment_status,
                    fee,
                    CASE
                        WHEN time_out IS NULL THEN NOW()
                        ELSE time_out
                    END as calc_time_out,
                    CASE
                        WHEN time_out IS NULL THEN
                            3000 + (FLOOR(EXTRACT(EPOCH FROM (NOW() - time_in)) / 3600) * 5000)
                        ELSE COALESCE(fee, 0)
                    END as calc_fee
                FROM parking_sessions
                WHERE card_id = %s
                  AND payment_status = 'unpaid'
                ORDER BY session_id DESC
                LIMIT 1
            """, (card_id,))
            return cur.fetchone()
        except Exception as e:
            logging.error(f"[DATABASE] Failed to fetch unpaid session for {card_id}: {e}")
            return None
        finally:
            conn.close()

    def _sync_in_record_from_session(self, card_id: str, session: dict):
        """Đồng bộ _in_records với dữ liệu session từ database để auto-confirm."""
        if not card_id or not session:
            return

        plate = session.get('plate_number')
        time_in = session.get('time_in') or datetime.datetime.now()
        session_id = session.get('session_id')

        with self._rec_lock:
            record = self._in_records.get(card_id, {})
            record.update({
                "session_id": session_id,
                "plate": plate or record.get('plate') or '',
                "time": time_in,
                "card_id": card_id,
                "paid": False
            })
            self._in_records[card_id] = record

    def _check_card_valid(self, card_id: str) -> bool:
        """Kiểm tra thẻ có hợp lệ (có trong database) không"""
        if not card_id:
            return False

        try:
            # Gọi API web để kiểm tra thẻ (dùng server_url từ config)
            web_url = self.cfg.server_url
            response = requests.get(f"{web_url}/api/check_card/{card_id}", timeout=5)

            if response.status_code == 200:
                data = response.json()
                is_valid = data.get("valid", False)
                logging.info(f"[CARD CHECK] {card_id}: {'VALID' if is_valid else 'INVALID'}")
                return is_valid
            else:
                logging.warning(f"[CARD CHECK] API returned {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            logging.error("[CARD CHECK] Timeout khi kết nối API")
            return False
        except Exception as e:
            logging.error(f"[CARD CHECK] Error: {e}")
            return False

    def _build_ui(self):
        self.setWindowTitle("Phần mềm quản lý bãi gửi xe")
        self.resize(1280, 780)
        self.setMinimumSize(800, 600)  # Minimum size cho màn hình nhỏ

        act_settings = QAction("Thiết lập", self); act_settings.triggered.connect(self.open_settings)
        act_full = QAction("Toàn màn hình", self, checkable=True); act_full.triggered.connect(self.toggle_fullscreen)
        menu = self.menuBar().addMenu("Cài đặt"); menu.addAction(act_settings); menu.addAction(act_full)

        self.lbl_cam_in  = qlabel_video_placeholder()
        self.lbl_img_in  = qlabel_video_placeholder("Ảnh xe vào")
        self.lbl_cam_out = qlabel_video_placeholder()
        self.lbl_img_out = qlabel_video_placeholder("Ảnh xe ra")

        grid = QGridLayout()
        grid.addWidget(self._group("Camera ngõ vào", self.lbl_cam_in), 0, 0)
        grid.addWidget(self._group("Ảnh xe vào", self.lbl_img_in),     0, 1)
        grid.addWidget(self._group("Camera ngõ ra", self.lbl_cam_out), 1, 0)
        grid.addWidget(self._group("Ảnh xe ra", self.lbl_img_out),     1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1);    grid.setRowStretch(1, 1)
        left = QWidget(); left.setLayout(grid)

        self.lbl_clock = QLabel("--:--:--"); self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet("font-size:22px;font-weight:600;")

        self.lbl_mqtt_state = QLabel("OFF"); self.lbl_mqtt_state.setStyleSheet("color:#bbb;font-weight:700;")
        self.lbl_mqtt_broker = QLabel("-"); self.lbl_mqtt_gate = QLabel("-"); self.lbl_mqtt_cid = QLabel("-")
        self.lbl_esp_last_msg = QLabel("-")
        self.lbl_esp_devices = QLabel("Không có thiết bị")  # Hiển thị danh sách ESP32
        self.lbl_esp_devices.setWordWrap(True)
        self.lbl_esp_devices.setStyleSheet("background:#2a2a2a;color:#ddd;padding:8px;border:1px solid #3a3a3a;border-radius:4px;")

        mqtt_form = QFormLayout()
        mqtt_form.addRow("Trạng thái:", self.lbl_mqtt_state)
        mqtt_form.addRow("Broker:", self.lbl_mqtt_broker)
        mqtt_form.addRow("Gate ID:", self.lbl_mqtt_gate)
        mqtt_form.addRow("Client ID:", self.lbl_mqtt_cid)
        mqtt_form.addRow("Tin nhắn cuối:", self.lbl_esp_last_msg)

        devices_label = QLabel("Thiết bị ESP32:")
        devices_label.setStyleSheet("font-weight:600;margin-top:8px;")

        mqtt_vbox = QVBoxLayout()
        form_widget = QWidget(); form_widget.setLayout(mqtt_form)
        mqtt_vbox.addWidget(form_widget)
        mqtt_vbox.addWidget(devices_label)
        mqtt_vbox.addWidget(self.lbl_esp_devices)

        box_mqtt = QGroupBox("Kết nối MQTT / ESP32")
        box_mqtt.setLayout(mqtt_vbox)


        self.ed_plate_cnt = self._count_box("0")
        self.ed_card  = self._ro_edit()
        self.ed_plate = self._ro_edit()
        self.ed_tin   = self._ro_edit()
        self.ed_tout  = self._ro_edit()
        self.ed_tdiff = self._ro_edit()
        self.ed_fee   = self._ro_edit()
        self.ed_slots_total = self._ro_edit()
        self.ed_slots_used  = self._ro_edit()
        self.ed_slots_free  = self._ro_edit()
        self.ed_total_revenue = self._ro_edit()  # ✅ NEW: Tổng tiền thu được

        # ✅ Chỉ giữ nút Xóa (các nút chụp đã được thay thế bằng MQTT tự động)
        btn_clear = QPushButton("Xóa");     btn_clear.clicked.connect(self.on_clear)

        # ✅ Nút đăng xuất
        btn_logout = QPushButton("Đăng xuất")
        btn_logout.clicked.connect(self.on_logout)
        btn_logout.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)

        form = QGridLayout(); r=0
        for label, widget in [
            ("SỐ XE", self.ed_plate_cnt),
            ("MÃ THẺ", self.ed_card),
            ("BIỂN SỐ", self.ed_plate),
            ("T/G XE VÀO", self.ed_tin),
            ("T/G XE RA", self.ed_tout),
            ("T/G GỬI XE", self.ed_tdiff),
            ("PHÍ GỬI XE", self.ed_fee),
            ("SLOT TỔNG", self.ed_slots_total),
            ("ĐÃ ĐỖ", self.ed_slots_used),
            ("CÒN LẠI", self.ed_slots_free),
            ("TỔNG TIỀN", self.ed_total_revenue),  # ✅ NEW
        ]:
            form.addWidget(QLabel(label), r, 0); form.addWidget(widget, r, 1); r += 1
        form.addWidget(btn_clear, r, 0, 1, 2)  # Nút xóa chiếm 2 cột
        r += 1
        form.addWidget(btn_logout, r, 0, 1, 2)  # Nút đăng xuất chiếm 2 cột

        box_info = QGroupBox("Thông tin"); wi = QWidget(); wi.setLayout(form)
        lay_info = QVBoxLayout(); lay_info.addWidget(wi); box_info.setLayout(lay_info)

        right = QVBoxLayout()
        right.addWidget(self.lbl_clock); right.addWidget(box_mqtt); right.addWidget(box_info); right.addStretch(1)
        panel_right = QWidget(); panel_right.setLayout(right)
        panel_right.setMinimumWidth(280)  # Minimum width thay vì maximum
        panel_right.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Thêm ScrollArea cho panel phải để responsive trên màn hình nhỏ
        scroll_right = QScrollArea()
        scroll_right.setWidget(panel_right)
        scroll_right.setWidgetResizable(True)
        scroll_right.setMinimumWidth(300)
        scroll_right.setMaximumWidth(500)
        scroll_right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        central = QWidget(); h = QHBoxLayout(central); h.addWidget(left, 3); h.addWidget(scroll_right, 1)
        self.setCentralWidget(central)

        # ✅ Status bar với thông tin nhân viên
        sb = QStatusBar()
        self.lbl_status_cam = QLabel("Camera: —")
        sb.addWidget(self.lbl_status_cam)

        # ✅ Thêm thông tin nhân viên đang đăng nhập
        self.lbl_staff_info = QLabel("")
        self.lbl_staff_info.setStyleSheet("color: #17a2b8; font-weight: bold; margin-left: 20px;")
        sb.addPermanentWidget(self.lbl_staff_info)

        self.setStatusBar(sb)

        self.ed_slots_total.setText(str(load_config().total_slots))
        self._update_slot_counts()
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)

        # ✅ Tạo label cảnh báo thẻ không hợp lệ (overlay)
        self.lbl_invalid_card_warning = QLabel(self)
        self.lbl_invalid_card_warning.setAlignment(Qt.AlignCenter)
        self.lbl_invalid_card_warning.setStyleSheet(
            "background-color:#dc3545;"
            "color:white;"
            "font-size:24pt;"
            "font-weight:bold;"
            "padding:30px;"
            "border:5px solid #ff0000;"
            "border-radius:10px;"
        )
        self.lbl_invalid_card_warning.setWordWrap(True)
        # Set geometry ban đầu (sẽ được update lại khi show)
        self.lbl_invalid_card_warning.setGeometry(340, 240, 600, 300)
        self.lbl_invalid_card_warning.hide()  # Ẩn mặc định

        # Timer để auto-hide warning
        self.timer_hide_warning = QTimer(self)
        self.timer_hide_warning.setSingleShot(True)
        self.timer_hide_warning.timeout.connect(self.lbl_invalid_card_warning.hide)

    def _load_data_from_db(self):
        """✅ Load dữ liệu từ PostgreSQL database"""
        conn = self._get_db_connection()
        if not conn:
            logging.error("[DATABASE] Cannot load data - no connection")
            return

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Load xe đang đậu (chưa ra - time_out IS NULL)
            cur.execute("""
                SELECT session_id, card_id, plate_number, time_in, payment_status
                FROM parking_sessions
                WHERE time_out IS NULL
                ORDER BY time_in DESC
            """)
            sessions = cur.fetchall()

            for session in sessions:
                card_id = session['card_id']
                if card_id:  # Chỉ load nếu có card_id
                    self._in_records[card_id] = {
                        "session_id": session['session_id'],
                        "plate": session['plate_number'],
                        "time": session['time_in'],
                        "card_id": card_id,
                        "paid": session['payment_status'] == 'paid'
                    }

            # ✅ DEBUG: Log loaded cards
            logging.info(f"[DEBUG] Loaded {len(self._in_records)} active sessions from database")
            if self._in_records:
                sample_cards = list(self._in_records.keys())[:3]
                logging.info(f"[DEBUG] Sample card IDs: {sample_cards}")

            # Load tổng doanh thu hôm nay
            cur.execute("""
                SELECT COALESCE(SUM(fee), 0) AS total
                FROM parking_sessions
                WHERE DATE(time_in) = CURRENT_DATE AND payment_status = 'paid'
            """)
            result = cur.fetchone()
            self._total_revenue = int(result['total']) if result else 0

            # Load số lượng xe vào hôm nay
            cur.execute("""
                SELECT COUNT(*) AS count
                FROM parking_sessions
                WHERE DATE(time_in) = CURRENT_DATE
            """)
            result = cur.fetchone()
            self._total_in_count = result['count'] if result else 0

            logging.info(f"[DATABASE] Loaded: vehicles={len(self._in_records)}, revenue={self._total_revenue:,}, count={self._total_in_count}")

            # ✅ Update UI với dữ liệu đã load
            self.ed_total_revenue.setText(f"{self._total_revenue:,}")
            self.ed_plate_cnt.setText(str(self._total_in_count))
            self._update_slot_counts()

        except Exception as e:
            logging.error(f"[DATABASE] Load error: {e}", exc_info=True)
        finally:
            conn.close()

    def _save_data_to_db(self):
        """✅ Không cần save - dữ liệu đã được lưu trực tiếp vào PostgreSQL"""
        # Hàm này giữ lại để tương thích, nhưng không làm gì
        # Vì dữ liệu đã được lưu vào database trong on_shoot_in và on_shoot_out
        pass

    def _check_midnight(self):
        """✅ Kiểm tra và reset daily counter vào 00:00 (DEPRECATED - dữ liệu từ database)"""
        # Không cần nữa vì dữ liệu được load từ PostgreSQL database
        # _load_data_from_db() sẽ tự động load đúng số liệu hôm nay
        pass

    def _update_work_time(self):
        """Cập nhật thời gian làm việc trong status bar"""
        if self.current_user and self.login_time:
            duration_str = self.get_work_duration()
            staff_text = f"👤 {self.current_user['full_name'] or self.current_user['username']} ({self.current_user['role'].upper()}) - Ca: {duration_str}"
            self.lbl_staff_info.setText(staff_text)

    def _send_heartbeat(self):
        """Gửi heartbeat lên server để báo 'Tôi còn online'"""
        if hasattr(self, 'staff_session_id') and self.staff_session_id:
            try:
                conn = self._get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE staff_sessions
                        SET last_heartbeat = NOW()
                        WHERE session_id = %s AND is_active = true
                    """, (self.staff_session_id,))
                    conn.commit()
                    cur.close()
                    conn.close()
                    logging.debug(f"[HEARTBEAT] Sent for session {self.staff_session_id}")
            except Exception as e:
                logging.error(f"[HEARTBEAT] Failed: {e}")

    def _start_timers(self):
        self.tmr = QTimer(self); self.tmr.timeout.connect(self._tick); self.tmr.start(1000)
        self.tmr_hb = QTimer(self); self.tmr_hb.timeout.connect(self._check_esp_alive); self.tmr_hb.start(500)  # Kiểm tra mỗi 0.5 giây

        # ✅ NEW: Timer kiểm tra midnight
        self.tmr_midnight = QTimer(self)
        self.tmr_midnight.timeout.connect(self._check_midnight)
        self.tmr_midnight.start(60000)  # Check mỗi phút

        # ✅ Timer cập nhật thời gian làm việc
        self.tmr_work_time = QTimer(self)
        self.tmr_work_time.timeout.connect(self._update_work_time)
        self.tmr_work_time.start(60000)  # Cập nhật mỗi phút

        # ✅ Timer heartbeat để báo server "Tôi còn online"
        self.tmr_heartbeat = QTimer(self)
        self.tmr_heartbeat.timeout.connect(self._send_heartbeat)
        self.tmr_heartbeat.start(30000)  # Gửi heartbeat mỗi 30 giây

    def _group(self, title, widget):
        gb = QGroupBox(title); v = QVBoxLayout(); v.setContentsMargins(6,8,6,6); v.addWidget(widget); gb.setLayout(v)
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); return gb

    def _ro_edit(self):
        e = QLineEdit(); e.setReadOnly(True)
        e.setStyleSheet("QLineEdit{background:#2a2a2a;color:#ddd;padding:6px;border:1px solid #3a3a3a;}")
        return e

    def _count_box(self, val="0"):
        e = QLineEdit(val); e.setReadOnly(True); e.setAlignment(Qt.AlignCenter)
        e.setStyleSheet("QLineEdit{background:#39d353;color:#0a0a0a;font-size:18px;border-radius:6px;padding:6px;font-weight:700;}")
        return e

    def _tick(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S  —  %a, %d/%m/%Y"))

    def _is_full(self) -> bool:
        try:
            total = int(self.cfg.total_slots)
        except Exception:
            total = 0
        return len(self._in_records) >= total if total > 0 else False

    # ----------------------------------------------------------------------------------------------
    # MODEL/OCR
    # ----------------------------------------------------------------------------------------------
    def _init_models(self):
        try:
            logging.info("Initializing ALPR models...")
            self.alpr = ALPR(YOLO_MODEL_PATH, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, 
                           max_workers=4, cache_ttl=5.0, max_cache_size=100)
            logging.info("ALPR initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ALPR: {e}", exc_info=True)
            self.alpr = None
            QMessageBox.critical(self, "ALPR", f"Không khởi tạo được YOLO/EasyOCR:\n{e}")

    # ----------------------------------------------------------------------------------------------
    # CAMERA
    # ----------------------------------------------------------------------------------------------
    def start_cameras(self):
        logging.info("Starting cameras...")
        self.stop_cameras()
        
        if self.cfg.cam_in_index >= 0:
            try:
                self.cam_in_worker = CameraWorker(self.cfg.cam_in_index, mirror=False)
                self.cam_in_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_in, img))
                self.cam_in_worker.opened.connect(lambda ok: self._cam_status(ok, "IN", self.cfg.cam_in_index))
                self.cam_in_worker.error_occurred.connect(lambda err: logging.error(f"Camera IN: {err}"))
                self.cam_in_worker.start()
                logging.info(f"Camera IN (index {self.cfg.cam_in_index}) started")
            except Exception as e:
                logging.error(f"Failed to start camera IN: {e}")
                self.lbl_status_cam.setText("Camera IN: Lỗi khi mở")
        else:
            self.lbl_status_cam.setText("Camera IN: tắt")

        if self.cfg.cam_out_index >= 0:
            try:
                self.cam_out_worker = CameraWorker(self.cfg.cam_out_index, mirror=False)
                self.cam_out_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_out, img))
                self.cam_out_worker.opened.connect(lambda ok: self._cam_status(ok, "OUT", self.cfg.cam_out_index))
                self.cam_out_worker.error_occurred.connect(lambda err: logging.error(f"Camera OUT: {err}"))
                self.cam_out_worker.start()
                logging.info(f"Camera OUT (index {self.cfg.cam_out_index}) started")
            except Exception as e:
                logging.error(f"Failed to start camera OUT: {e}")
                self.lbl_status_cam.setText(self.lbl_status_cam.text() + " | OUT: Lỗi khi mở")
        else:
            cur = self.lbl_status_cam.text()
            self.lbl_status_cam.setText((cur + " | OUT: tắt") if cur and "—" not in cur else "Camera OUT: tắt")

    def stop_cameras(self):
        """✅ FIXED: Proper camera cleanup"""
        logging.info("Stopping cameras...")
        
        if getattr(self, "cam_in_worker", None):
            try:
                self.cam_in_worker.stop()
                logging.info("Camera IN stopped")
            except Exception as e:
                logging.error(f"Error stopping camera IN: {e}")
            self.cam_in_worker = None
            
        if getattr(self, "cam_out_worker", None):
            try:
                self.cam_out_worker.stop()
                logging.info("Camera OUT stopped")
            except Exception as e:
                logging.error(f"Error stopping camera OUT: {e}")
            self.cam_out_worker = None
        
        # ✅ Wait for camera resources to be fully released
        QThread.msleep(500)

    def _cam_status(self, ok: bool, tag: str, idx: int):
        status = f"Camera {tag} (index {idx}): {'OK' if ok else 'Lỗi'}"
        self.lbl_status_cam.setText(status)
        logging.info(status)

    # ----------------------------------------------------------------------------------------------
    # SLOT / RECORD
    # ----------------------------------------------------------------------------------------------
    def _update_slot_counts(self):
        used = len(self._in_records)
        total = int(self.cfg.total_slots)
        free = max(0, total - used)
        self.ed_slots_used.setText(str(used))
        self.ed_slots_free.setText(str(free))
        self.ed_slots_total.setText(str(total))

    def _ensure_alpr(self) -> bool:
        if self.alpr is None:
            QMessageBox.warning(self, "ALPR", "Model YOLO/EasyOCR chưa sẵn sàng.")
            return False
        return True

    # ----------------------------------------------------------------------------------------------
    # LƯU ẢNH
    # ----------------------------------------------------------------------------------------------
    def _save_image_with_plate(self, plate: str, frame: np.ndarray, is_in: bool):
        root = DIR_IN if is_in else DIR_OUT
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = root / today
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_plate = plate.replace(" ", "_")
        path = str(save_dir / f"{safe_plate}.jpg")
        cv2.imwrite(path, frame)
        logging.info(f"Saved image: {path}")

    # ----------------------------------------------------------------------------------------------
    # HÀNH ĐỘNG: CHỤP IN / OUT (✅ FIXED: Thread safety)
    # ----------------------------------------------------------------------------------------------
    def _handle_shoot_in(self, card_id: str):
        """Handler for shoot in signal - runs in main thread"""
        logging.info(f"[_handle_shoot_in] Received signal with card: {card_id}")
        self._pending_card_id = card_id
        self.on_shoot_in()

    def _handle_shoot_out(self, card_id: str):
        """Handler for shoot out signal - runs in main thread"""
        logging.info(f"[_handle_shoot_out] Received signal with card: {card_id}")
        self._pending_card_id = card_id
        self.on_shoot_out()

    def _refresh_revenue_from_db(self):
        """✅ Query tổng doanh thu từ database và update UI"""
        conn = self._get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT COALESCE(SUM(fee), 0) AS total
                    FROM parking_sessions
                    WHERE DATE(time_in) = CURRENT_DATE AND payment_status = 'paid'
                """)
                result = cur.fetchone()
                total_revenue = int(result[0]) if result else 0
                self.ed_total_revenue.setText(f"{total_revenue:,}")
                logging.info(f"[REVENUE] Refreshed from database: {total_revenue:,} VND")
                return total_revenue
            except Exception as e:
                logging.error(f"[REVENUE] Query error: {e}")
                return 0
            finally:
                conn.close()
        return 0

    def _handle_update_revenue(self, card_id: str, plate: str, fee: int, total_revenue: int):
        """✅ NEW: Handler for revenue update signal - runs in main thread"""
        logging.info(f"[_handle_update_revenue] Updating UI: card={card_id}")
        self.ed_card.setText(card_id)
        self.ed_plate.setText(plate)
        self.ed_fee.setText(f"{fee:,}")
        # ✅ Query doanh thu từ database thay vì dùng biến local
        self._refresh_revenue_from_db()
        logging.info(f"✅ UI UPDATED VIA SIGNAL: Card={card_id}")

    def _handle_invalid_card(self, card_id: str):
        """✅ NEW: Handler for invalid card warning - runs in main thread"""
        logging.warning(f"[_handle_invalid_card] Showing warning for card: {card_id}")

        # Cập nhật text
        self.lbl_invalid_card_warning.setText(
            f"❌ THẺ KHÔNG HỢP LỆ!\n\n"
            f"Mã thẻ: {card_id}\n\n"
            f"Thẻ này không có trong hệ thống\n"
            f"hoặc đã bị vô hiệu hóa"
        )

        # Tính toán vị trí giữa màn hình
        window_rect = self.rect()
        label_width = 600
        label_height = 300
        x = (window_rect.width() - label_width) // 2
        y = (window_rect.height() - label_height) // 2

        self.lbl_invalid_card_warning.setGeometry(x, y, label_width, label_height)
        self.lbl_invalid_card_warning.raise_()  # Đưa lên trên cùng
        self.lbl_invalid_card_warning.show()

        # Auto-hide sau 4 giây
        self.timer_hide_warning.start(4000)

    def _initiate_sepay_qr_payment(
        self,
        gate_id: str,
        card_id: str,
        plate_number: str,
        session_id: Optional[int],
        amount: int,
        time_in: Optional[datetime.datetime],
    ):
        """Trigger SePay QR generation and publish to MQTT."""
        if not session_id:
            logging.warning(f"[SEPAY] Missing session_id for card {card_id}, cannot generate QR")
            return

        time_in_str = time_in.strftime("%H:%M:%S") if isinstance(time_in, datetime.datetime) else ""
        try:
            qr_result = create_parking_payment_sepay(session_id=session_id, amount=amount)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error(f"[SEPAY] Unexpected error generating QR: {exc}")
            return

        if not qr_result.get("success"):
            logging.error(f"[SEPAY] Failed to generate QR: {qr_result.get('error')}")
            self.lbl_esp_last_msg.setText("Không tạo được QR thanh toán")
            return

        qr_payload = {
            "session_id": session_id,
            "card_id": card_id,
            "plate_number": plate_number,
            "amount": amount,
            "qr_url": qr_result.get("qr_url", ""),
            "qr_content": qr_result.get("qr_content", ""),
            "qr_base64": qr_result.get("qr_base64", ""),
            "time_in": time_in_str,
            "bank_name": qr_result.get("bank_name") or SEPAY_CONFIG.get("bank_short_name"),
            "bank_display_name": qr_result.get("bank_display_name") or SEPAY_CONFIG.get("bank_display_name"),
            "account_number": qr_result.get("account_number") or SEPAY_CONFIG.get("account_number"),
            "account_name": qr_result.get("account_name") or SEPAY_CONFIG.get("account_name"),
        }

        topic = f"parking/payment/{gate_id}/qr_data"
        try:
            self.mqtt_client.publish(topic, json.dumps(qr_payload))
            logging.info(f"[SEPAY] Published QR to {topic}: {plate_number} - {amount:,} VND")
        except Exception as exc:  # pylint: disable=broad-except
            logging.error(f"[SEPAY] Failed to publish QR payload: {exc}")

        # Update UI immediately
        self.trigger_display_qr.emit(qr_payload)

    def _handle_display_qr(self, payment_data: dict):
        """✅ NEW: Handler for QR payment display - runs in main thread (thread-safe)"""
        try:
            logging.info(f"[_handle_display_qr] Displaying QR for {payment_data.get('plate_number')}")

            resolved_session_id = self._resolve_payment_session_id(payment_data)
            if not resolved_session_id:
                logging.error(f"[_handle_display_qr] Missing session_id for payment data: {payment_data}")
                QMessageBox.warning(
                    self,
                    "Không tìm thấy phiên gửi xe",
                    "Không xác định được phiên gửi xe để theo dõi thanh toán.\n"
                    "Vui lòng kiểm tra lại thẻ hoặc biển số."
                )
                return

            payment_data['session_id'] = resolved_session_id
            payment_data['amount'] = self._normalize_fee_value(payment_data.get('amount'))

            # ✅ Fill missing bank/account fields with config defaults
            if not payment_data.get('bank_name'):
                payment_data['bank_name'] = SEPAY_CONFIG.get("bank_short_name")
            if not payment_data.get('bank_display_name'):
                payment_data['bank_display_name'] = SEPAY_CONFIG.get("bank_display_name") or payment_data['bank_name']
            if not payment_data.get('account_number'):
                payment_data['account_number'] = SEPAY_CONFIG.get("account_number")
            if not payment_data.get('account_name'):
                payment_data['account_name'] = SEPAY_CONFIG.get("account_name")

            # Tạo dialog nếu chưa tồn tại
            if not hasattr(self, 'qr_dialog'):
                self.qr_dialog = QRPaymentWidget(self)
                self.qr_dialog.payment_cancelled.connect(self._on_qr_payment_cancelled)

            # Hiển thị QR - bây giờ đã ở main thread nên an toàn
            self.qr_dialog.display_payment(payment_data)
            logging.info(f"✅ QR DISPLAYED VIA SIGNAL: {payment_data.get('plate_number')}")

            # ✅ START AUTO-CONFIRM POLLING với amount
            amount = self._normalize_fee_value(payment_data.get('amount', 0))
            self._start_payment_polling(resolved_session_id, amount)

        except Exception as e:
            logging.error(f"[_handle_display_qr] Error: {e}")

    @staticmethod
    def _normalize_fee_value(value) -> int:
        """✅ NEW: Convert any fee/amount value to a safe non-negative integer.
        Handles int, float, Decimal (from PostgreSQL), and string types.
        """
        try:
            if value is None:
                return 0
            # ✅ FIX: Include Decimal type from PostgreSQL
            if isinstance(value, (int, float, Decimal)):
                return max(0, int(value))
            if isinstance(value, str):
                cleaned = value.replace(",", "").strip()
                if not cleaned:
                    return 0
                return max(0, int(float(cleaned)))
        except (ValueError, TypeError):
            logging.warning(f"[Payment] Invalid fee value: {value}")
        return 0

    def _resolve_payment_session_id(self, payment_data: dict) -> Optional[int]:
        """✅ NEW: Ensure session_id is always available for QR payment tracking."""
        session_id = payment_data.get('session_id')
        try:
            if session_id is not None:
                return int(session_id)
        except (TypeError, ValueError):
            logging.warning(f"[Payment] Invalid session_id format: {session_id}")

        with self._rec_lock:
            card_id = payment_data.get('card_id')
            if card_id:
                record = self._in_records.get(card_id)
                if record and record.get('session_id'):
                    logging.info(f"[Payment] Resolved session via card_id {card_id}")
                    return record.get('session_id')

            plate_number = payment_data.get('plate_number')
            if plate_number:
                plate_upper = plate_number.upper()
                for record in self._in_records.values():
                    rec_plate = record.get('plate')
                    if rec_plate and rec_plate.upper() == plate_upper:
                        logging.info(f"[Payment] Resolved session via plate {plate_number}")
                        return record.get('session_id')

        return None

    # ✅ NEW: PAYMENT POLLING METHODS
    def _start_payment_polling(self, session_id: int, amount: int = 0):
        """✅ NEW: Start polling database để check payment status"""
        try:
            logging.info(f"[Payment Polling] Starting for session {session_id}, amount={amount:,}")

            # Stop previous polling nếu có
            if self.payment_polling_timer:
                self.payment_polling_timer.stop()

            self.payment_polling_session_id = session_id
            self.payment_polling_amount = amount  # ✅ Lưu amount để verify với SePay
            self.payment_polling_count = 0  # Reset counter

            # Create timer
            self.payment_polling_timer = QTimer(self)
            self.payment_polling_timer.timeout.connect(
                lambda: self._check_payment_status(session_id)
            )

            # Start polling mỗi 1 giây
            self.payment_polling_timer.start(self.payment_polling_interval)

        except Exception as e:
            logging.error(f"[Payment Polling] Error starting: {e}")

    def _stop_payment_polling(self):
        """✅ NEW: Stop polling database"""
        if self.payment_polling_timer:
            self.payment_polling_timer.stop()
            self.payment_polling_timer = None
            self.payment_polling_session_id = None
            logging.info("[Payment Polling] Stopped")

    def _check_payment_status(self, session_id: int):
        """✅ NEW: Check payment status từ database + SePay API fallback"""
        try:
            conn = self._get_db_connection()
            if not conn:
                logging.warning(f"[Polling] No DB connection for session {session_id}")
                return

            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT
                    payment_status,
                    payment_method,
                    fee,
                    plate_number,
                    session_id
                FROM parking_sessions
                WHERE session_id = %s
            """, (session_id,))

            result = cur.fetchone()

            if result:
                payment_status = result.get('payment_status')
                plate_number = result.get('plate_number')
                amount = self._normalize_fee_value(result.get('fee'))
                logging.info(f"[Polling] Session {session_id}: payment_status = '{payment_status}'")

                # ✅ Nếu đã thanh toán trong DB → auto-confirm
                if payment_status == 'paid':
                    conn.close()
                    logging.info(f"💰 Payment confirmed in database for session {session_id}")

                    auto_confirm_data = {
                        'session_id': session_id,
                        'plate_number': plate_number,
                        'amount': amount,
                        'payment_method': result.get('payment_method'),
                        'from_database': True
                    }

                    self.trigger_payment_auto_confirmed.emit(session_id, auto_confirm_data)
                    self._stop_payment_polling()
                    return

                # ✅ FALLBACK: Check trực tiếp qua SePay API mỗi 5 giây (mỗi 5 lần poll)
                self.payment_polling_count += 1
                # Dùng self.payment_polling_amount thay vì DB fee (có thể = 0)
                verify_amount = self.payment_polling_amount if self.payment_polling_amount > 0 else amount
                if payment_status == 'unpaid' and verify_amount > 0 and self.payment_polling_count % 5 == 0:
                    # ✅ Tìm với nội dung SEVQR (giống QR content)
                    description = f"SEVQR BAI XE SESSION {session_id}"
                    logging.info(f"[Polling] Checking SePay API for session {session_id}, amount={verify_amount:,}...")
                    try:
                        tx = sepay.verify_payment(verify_amount, description)
                        if tx:
                            logging.info(f"💰 [SePay API] Payment found for session {session_id}: {tx}")

                            # Update database
                            cur.execute("""
                                UPDATE parking_sessions
                                SET payment_status = 'paid', payment_method = 'online'
                                WHERE session_id = %s AND payment_status = 'unpaid'
                            """, (session_id,))
                            conn.commit()
                            conn.close()

                            auto_confirm_data = {
                                'session_id': session_id,
                                'plate_number': plate_number,
                                'amount': verify_amount,
                                'payment_method': 'online',
                                'from_sepay_api': True
                            }

                            self.trigger_payment_auto_confirmed.emit(session_id, auto_confirm_data)
                            self._stop_payment_polling()
                            return
                    except Exception as e:
                        logging.warning(f"[Polling] SePay API check failed: {e}")

            conn.close()

        except Exception as e:
            logging.error(f"[Payment Polling] Error checking status: {e}")

    def _on_payment_auto_confirmed(self, session_id: int, data: dict):
        """✅ NEW: Handle auto-confirmed payment (từ MQTT hoặc polling)"""
        try:
            logging.info(f"✅ AUTO-CONFIRMING PAYMENT: {data}")

            # Stop polling nếu còn chạy
            self._stop_payment_polling()

            # Lấy card_id từ session_id
            card_id = None
            with self._rec_lock:
                for cid, record in self._in_records.items():
                    if record.get('session_id') == session_id:
                        card_id = cid
                        break

            if not card_id:
                logging.warning(f"❌ Card not found for session {session_id}")
                return

            with self._rec_lock:
                record = self._in_records.get(card_id)
                if not record:
                    logging.warning(f"❌ Record not found for card {card_id}")
                    return

                amount_from_data = self._normalize_fee_value(data.get('amount'))

                # Tính phí nếu chưa tính
                if amount_from_data == 0:
                    now = datetime.datetime.now()
                    mins = max(1, int((now - record['time']).total_seconds() // 60))
                    fee = calculate_parking_fee(mins)
                else:
                    fee = amount_from_data

                # Mark as paid
                if not record.get('paid'):
                    record['paid'] = True
                    self._paid_cards[card_id] = datetime.datetime.now()
                    self._total_revenue += fee

                    logging.info(f"✅ Payment processed: Card={card_id}, Plate={record['plate']}, Fee={fee:,}, Total={self._total_revenue:,}")

                # Cập nhật revenue UI signal
                self.trigger_update_revenue.emit(
                    card_id,
                    record['plate'],
                    fee,
                    self._total_revenue
                )

            # Đóng QR dialog
            if hasattr(self, 'qr_dialog') and self.qr_dialog.isVisible():
                self.qr_dialog.accept()

            # ✅ GỬI MQTT ĐẾN ESP32 TFT - Thông báo thanh toán thành công
            if self.mqtt_client and self.mqtt_client.is_connected():
                # Format đúng theo ESP32 expects: session_id, status, amount
                payment_confirmed_payload = {
                    "session_id": session_id,
                    "status": "paid",
                    "amount": fee,
                    "card_id": card_id,
                    "plate_number": data.get('plate_number') or record.get('plate')
                }
                # Gửi đến gate02 (payment terminal) - topic: payment_confirmed
                self.mqtt_client.publish("parking/payment/gate02/payment_confirmed", json.dumps(payment_confirmed_payload))
                logging.info(f"📤 Sent payment_confirmed to ESP32 TFT: {payment_confirmed_payload}")

            # Hiển thị thông báo thành công
            QMessageBox.information(self, "Thanh toán thành công",
                                  f"Thanh toán cho biển số {data.get('plate_number')} thành công!\n"
                                  f"Số tiền: {fee:,} VNĐ")

        except Exception as e:
            logging.error(f"[Auto-Confirm] Error: {e}")

    def on_shoot_in(self):
        """✅ FIXED: Full lock + proper error handling + RFID card support"""
        with self._rec_lock:  # ✅ Lock entire function
            logging.info("=== START SHOOT IN ===")

            # Lấy mã thẻ RFID nếu có
            card_id = self._pending_card_id
            self._pending_card_id = ""  # Reset
            logging.info(f"[DEBUG on_shoot_in] Card ID: {card_id}")

            if not self._ensure_alpr():
                logging.error("[DEBUG on_shoot_in] ALPR not ready")
                return
            if not self.cam_in_worker:
                logging.error("[DEBUG on_shoot_in] No camera IN worker")
                QMessageBox.warning(self, "Chụp IN", "Chưa có camera IN.")
                return

            logging.info("[DEBUG on_shoot_in] Starting frame capture...")

            if self._is_full():
                QMessageBox.warning(self, "BÃI ĐẦY", "SLOT đã đầy, không thể ghi nhận thêm xe vào.")
                return

            try:
                # ✅ Dùng VOTE_FRAMES_IN thay vì VOTE_FRAMES
                frames = self.cam_in_worker.get_recent_frames(VOTE_FRAMES_IN, MIN_SHARPNESS, VOTE_GAP_MS)
                if not frames:
                    f = self.cam_in_worker.best_recent_frame()
                    if f is None:
                        QMessageBox.warning(self, "Chụp IN", "Không lấy được khung hình rõ.")
                        return
                    frames = [f]

                plate, debug = self.alpr.infer_multi(frames)
                if debug is not None:
                    set_pixmap_fit_no_upscale(self.lbl_img_in, np_to_qimage(debug))

                if not plate:
                    self.ed_plate.setText("Không đọc được")
                    QMessageBox.information(self, "Chụp IN", "Không nhận dạng được biển số.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    (DIR_IN / "UNREAD").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(DIR_IN / "UNREAD" / f"UNREAD_{ts}.jpg"), frames[0])
                    logging.warning("No plate detected in IN")
                    return

                # Kiểm tra mã thẻ đã được sử dụng chưa
                if card_id and card_id in self._in_records:
                    existing = self._in_records[card_id]
                    QMessageBox.warning(self, "Thẻ đã sử dụng",
                        f"Mã thẻ {card_id} đã được dùng cho xe {existing['plate']}!\n"
                        f"Thời gian vào: {existing['time'].strftime('%H:%M:%S')}")
                    logging.warning(f"Duplicate card IN attempt: {card_id} (plate: {plate})")
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp IN", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                if self._is_full():
                    QMessageBox.warning(self, "BÃI ĐẦY", "SLOT đã đầy, không thể ghi nhận thêm xe vào.")
                    return

                # ✅ KIỂM TRA BIỂN SỐ TRÙNG trong database
                conn_check = self._get_db_connection()
                if conn_check:
                    try:
                        cur_check = conn_check.cursor()
                        cur_check.execute("""
                            SELECT card_id FROM parking_sessions
                            WHERE plate_number = %s AND time_out IS NULL
                        """, (plate,))
                        existing = cur_check.fetchone()

                        if existing:
                            existing_card = existing[0]
                            QMessageBox.warning(self, "⚠️ BIỂN SỐ TRÙNG",
                                f"Biển số {plate} đã có xe đậu trong bãi!\n\n"
                                f"🎫 Thẻ hiện tại: {existing_card}\n"
                                f"🎫 Thẻ đang quẹt: {card_id}\n\n"
                                f"⚠️ Không thể cho vào!\n"
                                f"Vui lòng kiểm tra lại thẻ hoặc biển số.")
                            logging.warning(f"⚠️ DUPLICATE PLATE: {plate} already in parking with card {existing_card}, blocking card {card_id}")
                            return
                    except Exception as e:
                        logging.error(f"[DATABASE] Check duplicate error: {e}")
                    finally:
                        conn_check.close()

                now = datetime.datetime.now()

                # ✅ INSERT vào PostgreSQL database
                conn = self._get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        staff_id = self.current_user['user_id'] if self.current_user else None
                        cur.execute("""
                            INSERT INTO parking_sessions (card_id, plate_number, time_in, payment_status, staff_id)
                            VALUES (%s, %s, %s, 'unpaid', %s)
                            RETURNING session_id
                        """, (card_id, plate, now, staff_id))
                        session_id = cur.fetchone()[0]
                        conn.commit()

                        # Lưu vào memory để tracking
                        self._in_records[card_id] = {
                            "session_id": session_id,
                            "plate": plate,
                            "time": now,
                            "card_id": card_id,
                            "paid": False
                        }

                        logging.info(f"[DATABASE] Recorded IN: session_id={session_id}, Card={card_id}, Plate={plate}")
                        logging.info(f"[DEBUG] Card saved to _in_records: '{card_id}' (len={len(card_id)}, repr={repr(card_id)})")
                    except Exception as e:
                        conn.rollback()
                        logging.error(f"[DATABASE] Insert error: {e}")
                        QMessageBox.critical(self, "Lỗi Database", f"Không thể lưu vào database: {e}")
                        return
                    finally:
                        conn.close()
                else:
                    QMessageBox.critical(self, "Lỗi", "Không thể kết nối database!")
                    return

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.clear(); self.ed_tdiff.clear(); self.ed_fee.setText("0")
                self._update_slot_counts()
                self._save_image_with_plate(plate, frames[0], True)

                # ✅ Update daily counter
                self._total_in_count += 1
                self.ed_plate_cnt.setText(str(self._total_in_count))

                logging.info(f"✅ Daily counter updated: {self._total_in_count} vehicles today")

            except Exception as e:
                logging.error(f"Error in on_shoot_in: {e}", exc_info=True)
                QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {e}")

    def on_shoot_out(self):
        """✅ FIXED: Full lock + proper error handling + RFID card support"""
        with self._rec_lock:  # ✅ Lock entire function
            logging.info("=== START SHOOT OUT ===")

            # Lấy mã thẻ RFID nếu có
            card_id = self._pending_card_id
            self._pending_card_id = ""  # Reset

            if not self._ensure_alpr():
                return
            if not self.cam_out_worker:
                QMessageBox.warning(self, "Chụp OUT", "Chưa có camera OUT.")
                return

            try:
                # ✅ Dùng VOTE_FRAMES_OUT thay vì VOTE_FRAMES
                frames = self.cam_out_worker.get_recent_frames(VOTE_FRAMES_OUT, MIN_SHARPNESS, VOTE_GAP_MS)
                if not frames:
                    f = self.cam_out_worker.best_recent_frame()
                    if f is None:
                        QMessageBox.warning(self, "Chụp OUT", "Không lấy được khung hình rõ.")
                        return
                    frames = [f]

                plate, debug = self.alpr.infer_multi(frames)
                if debug is not None:
                    set_pixmap_fit_no_upscale(self.lbl_img_out, np_to_qimage(debug))

                if not plate:
                    QMessageBox.information(self, "Chụp OUT", "Không nhận dạng được biển số.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    (DIR_OUT / "UNREAD").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(DIR_OUT / "UNREAD" / f"UNREAD_{ts}.jpg"), frames[0])
                    logging.warning("No plate detected in OUT")
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp OUT", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                # Tìm bản ghi IN theo mã thẻ
                if card_id not in self._in_records:
                    QMessageBox.warning(self, "Chụp OUT",
                        f"Không tìm thấy bản ghi IN cho thẻ {card_id}!\n"
                        f"Vui lòng kiểm tra hoặc thẻ chưa được quẹt vào.")
                    logging.warning(f"OUT without IN: Card={card_id}, Plate={plate}")
                    self._save_image_with_plate(plate, frames[0], False)
                    return

                # Lấy thông tin từ bản ghi IN (KHÔNG pop, giữ lại để UPDATE database)
                in_record = self._in_records[card_id]
                session_id = in_record["session_id"]
                plate_in = in_record["plate"]
                t_in = in_record["time"]
                already_paid = in_record.get("paid", False)  # ✅ Check nếu đã thanh toán

                now = datetime.datetime.now()
                logging.info(f"Recorded OUT: Card={card_id}, Plate IN={plate_in}, Plate OUT={plate} at {now}")

                # ✅ Cảnh báo BIỂN SỐ KHÔNG KHỚP - màu ĐỎ, CHẶN XE
                if plate != plate_in:
                    sim = plate_similarity(plate, plate_in)
                    if sim < 0.7:  # Chỉ cảnh báo nếu quá khác biệt
                        msg_box = QMessageBox(self)
                        msg_box.setIcon(QMessageBox.Critical)  # Icon đỏ
                        msg_box.setWindowTitle("🚨 BIỂN SỐ KHÔNG KHỚP")
                        msg_box.setText(
                            f"CẢNH BÁO: BIỂN SỐ KHÔNG KHỚP!\n\n"
                            f"🚗 Xe VÀO: {plate_in}\n"
                            f"🚗 Xe RA: {plate}\n"
                            f"🎫 Mã thẻ: {card_id}\n\n"
                            f"⛔ XE BỊ CHẶN - KHÔNG CHO RA!\n\n"
                            f"Vui lòng kiểm tra lại thẻ và biển số."
                        )
                        cancel_btn = msg_box.addButton("Hủy", QMessageBox.RejectRole)
                        cancel_btn.setDefault(True)  # Enter = Hủy

                        # ✅ Style màu ĐỎ nguy hiểm
                        msg_box.setStyleSheet("""
                            QMessageBox {
                                background-color: #2a2a2a;
                            }
                            QLabel {
                                color: #ff4444;
                                font-size: 14px;
                                min-width: 400px;
                                font-weight: bold;
                            }
                            QPushButton {
                                background-color: #dc3545;
                                color: white;
                                font-size: 14px;
                                font-weight: bold;
                                padding: 8px 16px;
                                border-radius: 4px;
                                min-width: 120px;
                            }
                            QPushButton:hover {
                                background-color: #c82333;
                            }
                        """)

                        msg_box.exec()

                        logging.warning(f"⛔ BLOCKED EXIT: Plate mismatch - IN={plate_in}, OUT={plate}, Card={card_id}, similarity={sim:.2f}")

                        # ✅ RETURN - không cho xe ra (không cần trả lại vì không pop)
                        return

                diff = now - t_in
                mins = max(1, int(diff.total_seconds() // 60))
                fee  = calculate_parking_fee(mins)

                # ✅ KIỂM TRA PAYMENT STATUS TỪ DATABASE
                # Query payment status thật từ database (không tin memory)
                conn_check = self._get_db_connection()
                payment_status_db = 'unpaid'  # Default
                payment_method_db = None

                if conn_check:
                    try:
                        cur_check = conn_check.cursor()
                        cur_check.execute(
                            "SELECT payment_status, payment_method, fee FROM parking_sessions WHERE session_id = %s",
                            (session_id,)
                        )
                        result = cur_check.fetchone()
                        if result:
                            payment_status_db = result[0] or 'unpaid'
                            payment_method_db = result[1]
                            db_fee = result[2]
                            logging.info(f"[DB CHECK] payment_status={payment_status_db}, method={payment_method_db}, fee={db_fee}")
                        conn_check.close()
                    except Exception as e:
                        logging.error(f"Error checking payment status: {e}")
                        conn_check.close()

                # ✅ CHƯA THANH TOÁN → THU TIỀN MẶT (RFID OUT ở bãi đỗ xe)
                if payment_status_db == 'unpaid' and not already_paid:
                    logging.info(f"💵 UNPAID vehicle at OUT - Collecting CASH payment")

                    # Thu tiền mặt trực tiếp
                    self._total_revenue += fee
                    payment_method = 'cash'

                    # Hiển thị popup xác nhận thu tiền
                    msg_box = QMessageBox(self)
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("💵 Thanh toán tiền mặt")
                    msg_box.setText(
                        f"🚗 Biển số: {plate_in}\n"
                        f"⏱️ Thời gian: {mins} phút\n"
                        f"💰 Số tiền: {fee:,} VNĐ\n\n"
                        f"➡️ Vui lòng thu tiền mặt"
                    )

                    # Nút OK - Đã thu tiền
                    ok_btn = msg_box.addButton("✅ Đã thu tiền", QMessageBox.AcceptRole)
                    ok_btn.setDefault(True)

                    # Style
                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #2a2a2a;
                        }
                        QLabel {
                            color: white;
                            font-size: 16px;
                            font-weight: bold;
                            min-width: 400px;
                        }
                        QPushButton {
                            background-color: #28a745;
                            color: white;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 12px 24px;
                            border-radius: 6px;
                            min-width: 180px;
                        }
                        QPushButton:hover {
                            background-color: #218838;
                        }
                    """)

                    msg_box.exec()

                    logging.info(f"💰 Cash payment collected: {fee:,} VND")

                # ✅ ĐÃ THANH TOÁN (QR hoặc Cash) → CHO RA
                elif payment_status_db == 'paid' or already_paid:
                    logging.info(f"✅ Vehicle already paid - Opening gate")

                    # Hiển thị thông báo
                    QMessageBox.information(
                        self,
                        "✅ Đã thanh toán",
                        f"Thanh toán thành công!\n\n"
                        f"🚗 Biển số: {plate_in}\n"
                        f"💳 Phương thức: {payment_method_db}\n\n"
                        f"Cổng đang mở..."
                    )
                    # Không cần popup nếu đã thanh toán online/cash khác

                # ✅ Xe đã thanh toán online - giữ nguyên fee đã thanh toán
                else:
                    # Query fee hiện tại từ database (đã được set khi thanh toán terminal)
                    conn_check = self._get_db_connection()
                    if conn_check:
                        try:
                            cur_check = conn_check.cursor()
                            cur_check.execute("SELECT fee, payment_method, payment_status FROM parking_sessions WHERE session_id = %s", (session_id,))
                            result = cur_check.fetchone()
                            if result:
                                db_fee = result[0]
                                db_method = result[1]
                                db_status = result[2]
                                logging.info(f"[DEBUG] Database values: fee={db_fee}, method={db_method}, status={db_status}")
                                fee = int(db_fee) if db_fee else calculate_parking_fee(mins)
                            else:
                                fee = calculate_parking_fee(mins)
                                logging.warning(f"[DEBUG] No database record found for session_id={session_id}")
                        except Exception as e:
                            logging.error(f"[DEBUG] Query fee error: {e}")
                            fee = calculate_parking_fee(mins)
                        finally:
                            conn_check.close()
                    else:
                        fee = calculate_parking_fee(mins)
                    logging.info(f"✅ Already paid via terminal, fee={fee:,} VND (no cash collection, no revenue update)")

                # ✅ UPDATE database - SET time_out, duration
                conn = self._get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()

                        # ✅ CRITICAL FIX: Check payment_status_db (from database) thay vì chỉ check already_paid (from memory)
                        # Nếu đã thanh toán (online/cash), chỉ update time_out và duration (GIỮ NGUYÊN fee và payment_method)
                        if payment_status_db == 'paid' or already_paid:
                            cur.execute("""
                                UPDATE parking_sessions
                                SET time_out = %s,
                                    duration_minutes = %s
                                WHERE session_id = %s
                            """, (now, mins, session_id))
                            logging.info(f"[DATABASE] Vehicle already paid - Updated time_out only (fee unchanged)")
                        # ✅ Nếu chưa thanh toán, update toàn bộ
                        else:
                            payment_method = 'cash'
                            cur.execute("""
                                UPDATE parking_sessions
                                SET time_out = %s,
                                    duration_minutes = %s,
                                    fee = %s,
                                    payment_status = 'paid',
                                    payment_method = %s
                                WHERE session_id = %s
                            """, (now, mins, fee, payment_method, session_id))
                        conn.commit()

                        # Xóa khỏi memory (xe đã ra)
                        del self._in_records[card_id]

                        # ✅ Log rõ ràng: Fee có bị update không?
                        if payment_status_db == 'paid' or already_paid:
                            logging.info(f"[DATABASE] Updated OUT (already paid): session_id={session_id}, time_out={now.strftime('%H:%M:%S')}, duration={mins}min")
                        else:
                            logging.info(f"[DATABASE] Updated OUT (cash payment): session_id={session_id}, fee={fee:,}, method=cash")
                    except Exception as e:
                        conn.rollback()
                        logging.error(f"[DATABASE] Update error: {e}")
                        QMessageBox.critical(self, "Lỗi Database", f"Không thể cập nhật database: {e}")
                    finally:
                        conn.close()
                else:
                    logging.error("[DATABASE] Cannot update - no connection")

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(t_in.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tdiff.setText(f"{mins} phút")
                self.ed_fee.setText(f"{fee:,}" if fee > 0 else "Đã thanh toán")
                # ✅ Query doanh thu từ database thay vì dùng biến local
                total_revenue = self._refresh_revenue_from_db()
                self._update_slot_counts()
                self._save_image_with_plate(plate, frames[0], False)

                logging.info(f"✅ Total revenue: {total_revenue:,} VND")

            except Exception as e:
                logging.error(f"Error in on_shoot_out: {e}", exc_info=True)
                QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {e}")

    # ----------------------------------------------------------------------------------------------
    # MISC
    # ----------------------------------------------------------------------------------------------
    def on_clear(self):
        for w in [self.ed_card, self.ed_plate, self.ed_tin, self.ed_tout, self.ed_tdiff, self.ed_fee]:
            try: w.clear()
            except Exception: pass
        for lbl, text in [(self.lbl_img_in,"Ảnh xe vào"), (self.lbl_img_out, "Ảnh xe ra")]:
            lbl.clear(); lbl.setText(text)

    def on_logout(self):
        """Xử lý đăng xuất - CHỈ cách duy nhất để thoát app"""
        reply = QMessageBox.question(
            self,
            "Xác nhận đăng xuất",
            f"Bạn có chắc muốn đăng xuất?\n\nNhân viên: {self.current_user.get('full_name', 'N/A')}\nThời gian làm việc: {self.get_work_duration()}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # ✅ Cập nhật logout_time vào database
            if hasattr(self, 'staff_session_id') and self.staff_session_id:
                try:
                    conn = self._get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE staff_sessions
                            SET logout_time = NOW(), is_active = false
                            WHERE session_id = %s
                        """, (self.staff_session_id,))
                        conn.commit()
                        cur.close()
                        conn.close()
                        logging.info(f"[SESSION] Closed staff session ID: {self.staff_session_id}")
                except Exception as e:
                    logging.error(f"[SESSION] Failed to close session: {e}")

            logging.info(f"[LOGOUT] User {self.current_user.get('username')} logged out")
            self.allow_close = True  # Cho phép đóng cửa sổ
            self.close()  # Đóng app

    def get_work_duration(self):
        """Tính thời gian làm việc từ lúc đăng nhập (dùng server time)"""
        if hasattr(self, 'staff_session_id') and self.staff_session_id:
            try:
                conn = self._get_db_connection()
                if conn:
                    cur = conn.cursor()
                    # ✅ Dùng server time để tính chính xác
                    cur.execute("""
                        SELECT EXTRACT(EPOCH FROM (NOW() - login_time)) as duration_seconds
                        FROM staff_sessions
                        WHERE session_id = %s
                    """, (self.staff_session_id,))
                    result = cur.fetchone()
                    cur.close()
                    conn.close()

                    if result:
                        duration = max(0, result[0])
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        return f"{hours}h {minutes}p"
            except Exception as e:
                logging.error(f"[DURATION] Failed to get work duration: {e}")

        return "0h 0p"

    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            cam_in, cam_out, slots, en_mqtt, host, port, gate = dlg.values()
            if cam_in == -1:
                QMessageBox.warning(self, "Cài đặt", "Ngõ vào phải chọn 1 camera hợp lệ."); return
            self.cfg.cam_in_index = int(cam_in)
            self.cfg.cam_out_index = int(cam_out)
            self.cfg.total_slots = max(1, slots)
            self.cfg.mqtt_enable = bool(en_mqtt)
            self.cfg.mqtt_host   = host
            self.cfg.mqtt_port   = int(port)
            self.cfg.gate_id     = gate
            self.cfg.auto_start_broker = False
            save_config(self.cfg)
            
            logging.info("Settings saved, restarting cameras and MQTT...")
            self.start_cameras()
            self._update_slot_counts()
            self.restart_mqtt()
            self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
            self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    def _set_mqtt_state(self, text, color="#bbb"):
        self.lbl_mqtt_state.setText(text)
        self.lbl_mqtt_state.setStyleSheet(f"color:{color};font-weight:700;")

    def _update_esp_devices_display(self):
        """Cập nhật hiển thị danh sách ESP32"""
        if not self._esp_devices:
            self.lbl_esp_devices.setText("Không có thiết bị")
            self.lbl_esp_devices.setStyleSheet("background:#2a2a2a;color:#888;padding:8px;border:1px solid #3a3a3a;border-radius:4px;")
            return

        lines = []
        online_count = 0
        for mac, info in self._esp_devices.items():
            status = "🟢 Online" if info.get("online", False) else "🔴 Offline"
            ip = info.get("ip", "N/A")
            last_seen = info.get("last_hb", 0)
            elapsed = int(time.time() - last_seen) if last_seen > 0 else 0

            if info.get("online", False):
                online_count += 1
                lines.append(f"{status} | MAC: {mac}\n   IP: {ip} | Heartbeat: {elapsed}s trước")
            else:
                lines.append(f"{status} | MAC: {mac}\n   Mất kết nối")

        text = "\n\n".join(lines)
        color = "#d4f4dd" if online_count > 0 else "#888"
        self.lbl_esp_devices.setText(text)
        self.lbl_esp_devices.setStyleSheet(
            f"background:#2a2a2a;color:{color};padding:8px;border:1px solid #3a3a3a;border-radius:4px;"
        )


    def _refresh_conn_badge(self):
        online_count = sum(1 for dev in self._esp_devices.values() if dev.get("online", False))
        total_count = len(self._esp_devices)

        mqtt_txt = "Đã kết nối" if self._mqtt_connected else "Mất kết nối"

        if total_count > 0:
            esp_txt = f"{online_count}/{total_count} Online"
        else:
            esp_txt = "Không có thiết bị"

        color = "#39d353" if (self._mqtt_connected and online_count > 0) else ("#f1c40f" if self._mqtt_connected else "#ff6b6b")
        self._set_mqtt_state(f"MQTT: {mqtt_txt} | ESP32: {esp_txt}", color)

    def _check_esp_alive(self):
        """Kiểm tra trạng thái tất cả ESP32"""
        if not self._mqtt_connected:
            # Khi mất kết nối MQTT, đánh dấu tất cả ESP32 offline
            for mac in self._esp_devices:
                if self._esp_devices[mac].get("online", False):
                    self._esp_devices[mac]["online"] = False
                    logging.warning(f"ESP32 {mac} offline (MQTT disconnected)")
            self._refresh_conn_badge()
            self._update_esp_devices_display()
            return

        now = time.time()
        updated = False
        for mac, info in self._esp_devices.items():
            last_hb = info.get("last_hb", 0)
            if last_hb > 0 and (now - last_hb) > self._hb_timeout:
                if info.get("online", False):
                    info["online"] = False
                    updated = True
                    logging.warning(f"ESP32 {mac} heartbeat timeout")

        if updated:
            self._refresh_conn_badge()
            self._update_esp_devices_display()

    def ensure_broker_running(self):
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)
        if not (self.cfg.mqtt_enable and self.cfg.auto_start_broker):
            return
        host = (self.cfg.mqtt_host or "").strip()
        local_ips = get_local_ips()
        if host not in local_ips:
            return
        probe_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        if is_port_open(probe_host, self.cfg.mqtt_port):
            logging.info("Broker already running")
            return
        exe, conf = self.cfg.broker_exe, self.cfg.broker_conf
        if not os.path.exists(exe) or not os.path.exists(conf):
            self._set_mqtt_state("Không thấy mosquitto/conf", "#ff6b6b"); return
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            exe_dir = os.path.dirname(exe) or None
            self._mosq_proc = subprocess.Popen(
                [exe, "-c", conf],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                creationflags=flags, cwd=exe_dir
            )
            self._set_mqtt_state("Đang khởi động broker…", "#f1c40f")
            logging.info("Started Mosquitto broker")
        except Exception as e:
            self._set_mqtt_state(f"Lỗi chạy broker: {e}", "#ff6b6b")
            logging.error(f"Failed to start broker: {e}")
            self._mosq_proc = None

    def init_mqtt(self):
        if not self.cfg.mqtt_enable or mqtt is None:
            self._mqtt_connected = False
            self._esp_online = False
            self._set_mqtt_state("OFF", "#bbb")
            return
        try:
            cid = "ui-" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
            self.lbl_mqtt_cid.setText(cid)
            self.mqtt_client = mqtt.Client(client_id=cid, protocol=mqtt.MQTTv311)
            try:
                self.mqtt_client.reconnect_delay_set(min_delay=0.5, max_delay=3)
            except Exception:
                pass
            def _on_connect(client, userdata, flags, rc, properties=None):
                self._mqtt_connected = (rc == 0)
                if rc == 0:
                    base = f"parking/gate/{self.cfg.gate_id}"
                    client.subscribe(base + "/event", qos=1)
                    client.subscribe(base + "/stats", qos=1)
                    client.subscribe(base + "/status", qos=1)
                    client.subscribe(base + "/heartbeat", qos=0)
                    client.subscribe(base + "/in", qos=1)
                    client.subscribe(base + "/out", qos=1)
                    client.subscribe(base + "/payment", qos=1)  # ✅ NEW: Payment confirmation

                    # ✅ Subscribe to payment terminal topics
                    client.subscribe("parking/payment/+/heartbeat", qos=0)
                    client.subscribe("parking/payment/+/status", qos=1)
                    client.subscribe("parking/payment/+/card_scanned", qos=1)
                    client.subscribe("parking/payment/+/qr_data", qos=1)  # ✅ NEW: QR code payment data
                    client.subscribe("parking/payment/+/payment_confirmed", qos=1)
                    client.subscribe("parking/thanhtoan/payment_success", qos=1)

                    logging.info("MQTT connected and subscribed to all topics")
                else:
                    # Đánh dấu tất cả thiết bị offline
                    for mac in self._esp_devices:
                        self._esp_devices[mac]["online"] = False
                    logging.error(f"MQTT connection failed: rc={rc}")
                self._refresh_conn_badge()
                self._update_esp_devices_display()
            def _on_disconnect(client, userdata, rc, properties=None):
                self._mqtt_connected = False
                # Đánh dấu tất cả thiết bị offline
                for mac in self._esp_devices:
                    self._esp_devices[mac]["online"] = False
                self._refresh_conn_badge()
                self._update_esp_devices_display()
                logging.warning("MQTT disconnected")
            def _on_message(client, userdata, msg):
                try:
                    topic = msg.topic
                    payload = {}
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except Exception:
                        pass
                    base = f"parking/gate/{self.cfg.gate_id}"

                    # Lấy MAC address từ payload
                    mac = payload.get("mac", None)

                    if topic.endswith("/status"):
                        online = bool(payload.get("online", False))
                        if mac:
                            # Cập nhật hoặc tạo mới thông tin ESP32
                            if mac not in self._esp_devices:
                                self._esp_devices[mac] = {}
                            self._esp_devices[mac]["online"] = online
                            self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                            self._esp_devices[mac]["last_hb"] = time.time() if online else 0
                            logging.info(f"ESP32 {mac} status: {'online' if online else 'offline'}")
                        self._refresh_conn_badge()
                        self._update_esp_devices_display()

                    elif topic.endswith("/heartbeat"):
                        if mac:
                            # Cập nhật heartbeat và thông tin
                            if mac not in self._esp_devices:
                                self._esp_devices[mac] = {}
                            self._esp_devices[mac]["online"] = True
                            self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                            self._esp_devices[mac]["last_hb"] = time.time()
                            self._refresh_conn_badge()
                            self._update_esp_devices_display()

                    elif topic == base + "/in":
                        card_id = payload.get("card_id", "")
                        logging.info(f"MQTT trigger: IN with card {card_id}")

                        # ✅ KIỂM TRA THẺ HỢP LỆ
                        if card_id and not self._check_card_valid(card_id):
                            logging.warning(f"❌ THẺ KHÔNG HỢP LỆ: {card_id}")
                            self.lbl_esp_last_msg.setText(f"⚠️ THẺ KHÔNG HỢP LỆ: {card_id}")

                            # Emit signal để hiển thị cảnh báo trong main thread (không lag)
                            self.trigger_invalid_card.emit(card_id)
                            return

                        msg_text = f"Yêu cầu chụp IN"
                        if card_id:
                            msg_text += f" | Thẻ: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
                        # Emit signal để trigger chụp trong main thread
                        logging.info(f"[DEBUG] Emitting trigger_shoot_in signal with card: {card_id}")
                        self.trigger_shoot_in.emit(card_id)

                    elif topic == base + "/out":
                        card_id = payload.get("card_id", "")
                        logging.info(f"MQTT trigger: OUT with card {card_id}")

                        # ✅ KIỂM TRA THẺ HỢP LỆ
                        if card_id and not self._check_card_valid(card_id):
                            logging.warning(f"❌ THẺ KHÔNG HỢP LỆ: {card_id}")
                            self.lbl_esp_last_msg.setText(f"⚠️ THẺ KHÔNG HỢP LỆ: {card_id}")

                            # Emit signal để hiển thị cảnh báo trong main thread (không lag)
                            self.trigger_invalid_card.emit(card_id)
                            return

                        msg_text = f"Yêu cầu chụp OUT"
                        if card_id:
                            msg_text += f" | Thẻ: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
                        # Emit signal để trigger chụp trong main thread
                        logging.info(f"[DEBUG] Emitting trigger_shoot_out signal with card: {card_id}")
                        self.trigger_shoot_out.emit(card_id)

                    elif topic == base + "/payment":
                        # ❌ DEPRECATED - Use /payment_confirmed instead
                        logging.warning(f"⚠️ Received deprecated topic: {topic}, use /payment_confirmed instead")
                        pass

                    elif topic == "parking/thanhtoan/payment_success":
                        session_id_val = payload.get("session_id")
                        try:
                            session_id_int = int(session_id_val)
                        except (TypeError, ValueError):
                            logging.warning(f"[Payment] Invalid session_id in payment_success payload: {session_id_val}")
                            session_id_int = None

                        if session_id_int is None:
                            return

                        auto_confirm_data = {
                            "session_id": session_id_int,
                            "plate_number": payload.get("plate_number"),
                            "amount": payload.get("amount"),
                            "payment_method": payload.get("method", "sepay"),
                            "reference_code": payload.get("reference_code"),
                            "source": "mqtt",
                        }
                        logging.info(f"[Payment] MQTT payment_success received: {auto_confirm_data}")
                        self.trigger_payment_auto_confirmed.emit(session_id_int, auto_confirm_data)

                    elif topic.endswith("/event"):
                        event_type = payload.get("type", "unknown")
                        msg_text = f"Event: {event_type}"
                        if mac:
                            msg_text += f" (từ {mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                    elif topic.endswith("/stats"):
                        msg_text = "Nhận thống kê"
                        if mac:
                            msg_text += f" (từ {mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                    # ✅ Payment terminal topics
                    elif "parking/payment/" in topic:
                        if topic.endswith("/heartbeat"):
                            if mac:
                                if mac not in self._esp_devices:
                                    self._esp_devices[mac] = {}
                                self._esp_devices[mac]["online"] = True
                                self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                                self._esp_devices[mac]["last_hb"] = time.time()
                                self._refresh_conn_badge()
                                self._update_esp_devices_display()

                        elif topic.endswith("/status"):
                            online = bool(payload.get("online", False))
                            if mac:
                                if mac not in self._esp_devices:
                                    self._esp_devices[mac] = {}
                                self._esp_devices[mac]["online"] = online
                                self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                                self._esp_devices[mac]["last_hb"] = time.time() if online else 0
                                logging.info(f"Payment Terminal {mac} status: {'online' if online else 'offline'}")
                            self._refresh_conn_badge()
                            self._update_esp_devices_display()

                        elif topic.endswith("/card_scanned"):
                            # Payment terminal đã quét thẻ, gửi thông tin xe về
                            card_id = (payload.get("card_id") or "").strip()
                            logging.info(f"💳 Payment terminal scanned card: {card_id}")

                            gate_id = topic.split("/")[2] if len(topic.split("/")) >= 3 else "unknown"
                            qr_request_payload = None

                            # ✅ ƯU TIÊN LOGIC GIỐNG WEB: lấy session trực tiếp từ database
                            db_session = self._fetch_latest_unpaid_session(card_id)
                            if db_session:
                                fee = self._normalize_fee_value(db_session.get('calc_fee') or db_session.get('fee'))
                                time_in_dt = db_session.get('time_in')
                                time_out_dt = db_session.get('calc_time_out')
                                time_in_str = time_in_dt.strftime("%H:%M:%S") if isinstance(time_in_dt, datetime.datetime) else ""
                                time_out_str = time_out_dt.strftime("%H:%M:%S") if isinstance(time_out_dt, datetime.datetime) else ""
                                plate_number = db_session.get('plate_number') or "UNKNOWN"

                                vehicle_info = {
                                    "plate": plate_number,
                                    "time_in": time_in_str,
                                    "time_out": time_out_str,
                                    "fee": fee,
                                    "paid": False
                                }

                                response_topic = f"parking/payment/{gate_id}/vehicle_info"
                                self.mqtt_client.publish(response_topic, json.dumps(vehicle_info))
                                logging.info(f"📤 Sent vehicle_info from DB: {vehicle_info}")
                                self.lbl_esp_last_msg.setText(f"Info {plate_number}: {fee:,} VNĐ")

                                qr_request_payload = {
                                    "gate_id": gate_id,
                                    "card_id": card_id,
                                    "plate_number": plate_number,
                                    "session_id": db_session.get("session_id"),
                                    "amount": fee,
                                    "time_in": time_in_dt,
                                }

                                # Đồng bộ in-memory records để auto-confirm sau này
                                self._sync_in_record_from_session(card_id, db_session)

                            else:
                                with self._rec_lock:
                                    # ✅ DEBUG: Log tất cả card IDs trong _in_records
                                    logging.info(f"[DEBUG] Card scanned: '{card_id}' (len={len(card_id)})")
                                    logging.info(f"[DEBUG] Cards in _in_records: {list(self._in_records.keys())}")

                                    if card_id in self._in_records:
                                        record = self._in_records[card_id]
                                        already_paid = record.get("paid", False)

                                        # ✅ Tính phí theo thời gian thực
                                        now = datetime.datetime.now()
                                        diff = now - record["time"]
                                        mins = max(1, int(diff.total_seconds() // 60))
                                        fee = 0 if already_paid else calculate_parking_fee(mins)
                                        
                                        # 🔍 DEBUG: Log trạng thái thanh toán
                                        logging.info(f"[DEBUG PAYMENT] Card: {card_id}, Already_paid: {already_paid}, Fee: {fee:,} VND, Duration: {mins} mins")

                                        # ✅ Nếu xe ĐÃ THANH TOÁN → Gửi vehicle_info (để hiển thị "Đã thanh toán")
                                        if already_paid:
                                            vehicle_info = {
                                                "plate": record["plate"],
                                                "time_in": record["time"].strftime("%H:%M:%S"),
                                                "time_out": now.strftime("%H:%M:%S"),
                                                "fee": fee,
                                                "paid": True
                                            }

                                            response_topic = f"parking/payment/{gate_id}/vehicle_info"
                                            self.mqtt_client.publish(response_topic, json.dumps(vehicle_info))
                                            logging.info(f"📤 Sent vehicle info (ALREADY PAID): {vehicle_info}")
                                            self.lbl_esp_last_msg.setText(f"Xe {record['plate']} đã thanh toán")

                                        # ✅ Nếu xe CHƯA THANH TOÁN → GỬI THÔNG TIN LÊN TFT
                                        else:
                                            logging.info(f"💳 Vehicle UNPAID - Sending vehicle info to TFT display")

                                            # ✅ CRITICAL FIX: GỬI vehicle_info TRƯỚC để ESP32 thoát vòng đợi!
                                            vehicle_info = {
                                                "plate": record["plate"],
                                                "time_in": record["time"].strftime("%H:%M:%S"),
                                                "time_out": now.strftime("%H:%M:%S"),
                                                "fee": fee,
                                                "paid": False  # Chưa thanh toán
                                            }
                                            response_topic = f"parking/payment/{gate_id}/vehicle_info"
                                            self.mqtt_client.publish(response_topic, json.dumps(vehicle_info))
                                            logging.info(f"📤 Sent vehicle_info (UNPAID) first: {vehicle_info}")

                                            logging.info(f"✅ Vehicle info sent to TFT: {record['plate']} - {fee:,} VNĐ")
                                            self.lbl_esp_last_msg.setText(f"Info {record['plate']}: {fee:,} VNĐ")

                                            qr_request_payload = {
                                                "gate_id": gate_id,
                                                "card_id": card_id,
                                                "plate_number": record["plate"],
                                                "session_id": record.get("session_id"),
                                                "amount": fee,
                                                "time_in": record["time"],
                                            }

                                    else:
                                        # ❌ Thẻ không tìm thấy → GỬI ERROR RESPONSE về ESP32!
                                        logging.warning(f"❌ Card {card_id} not found in records")

                                        # ✅ DEBUG: Kiểm tra case-insensitive và format khác
                                        found_similar = False
                                        for key in self._in_records.keys():
                                                if key.upper() == card_id.upper():
                                                    logging.warning(f"[DEBUG] FOUND CASE MISMATCH! Key in records: '{key}', Scanned: '{card_id}'")
                                                    found_similar = True
                                                    break

                                        if not found_similar:
                                            logging.warning(f"[DEBUG] No similar card found. Total records: {len(self._in_records)}")
                                            # Log first 5 records for comparison
                                            sample_keys = list(self._in_records.keys())[:5]
                                            logging.warning(f"[DEBUG] Sample keys: {sample_keys}")

                                        # Gửi error response
                                        error_info = {
                                            "plate": "KHONG TIM THAY",
                                            "time_in": "00:00:00",
                                            "time_out": "00:00:00",
                                            "fee": 0,
                                            "paid": False,
                                            "error": "Card not found in system"
                                        }

                                        response_topic = f"parking/payment/{gate_id}/vehicle_info"
                                        self.mqtt_client.publish(response_topic, json.dumps(error_info))
                                        logging.info(f"📤 Sent ERROR response for card: {card_id}")
                                        self.lbl_esp_last_msg.setText(f"Thẻ {card_id} không tìm thấy")

                            if qr_request_payload:
                                self._initiate_sepay_qr_payment(
                                    gate_id=qr_request_payload["gate_id"],
                                    card_id=qr_request_payload["card_id"],
                                    plate_number=qr_request_payload["plate_number"],
                                    session_id=qr_request_payload["session_id"],
                                    amount=qr_request_payload["amount"],
                                    time_in=qr_request_payload["time_in"],
                                )

                        elif topic.endswith("/qr_data"):
                            # ✅ NEW: Display QR code payment from server
                            logging.info(f"📱 QR Payment data received from MQTT")
                            try:
                                payment_data = {
                                    'session_id': payload.get('session_id'),
                                    'card_id': payload.get('card_id'),
                                    'plate_number': payload.get('plate_number'),
                                    'amount': payload.get('amount'),
                                    'qr_url': payload.get('qr_url', ''),
                                    'qr_content': payload.get('qr_content', ''),
                                    'qr_base64': payload.get('qr_base64', ''),
                                    'time_in': payload.get('time_in') or payload.get('entry_time')
                                }
                                logging.info(f"[QR] Displaying QR for {payment_data['plate_number']} - {payment_data['amount']:,} VNĐ")

                                # ✅ FIXED: Emit signal để hiển thị QR từ main thread (thread-safe)
                                self.trigger_display_qr.emit(payment_data)
                            except Exception as e:
                                logging.error(f"[QR] Error displaying QR: {e}")

                        elif topic.endswith("/payment_confirmed"):
                            # ✅ Payment terminal xác nhận thanh toán - CHỈ đánh dấu đã trả tiền
                            card_id = payload.get("card_id", "")
                            logging.info(f"💰 Payment confirmed from terminal: card={card_id}")

                            with self._rec_lock:
                                if card_id in self._in_records:
                                    record = self._in_records[card_id]  # ✅ KHÔNG pop(), chỉ lấy reference
                                    plate = record["plate"]
                                    session_id = record["session_id"]

                                    # ✅ Chỉ mark là đã thanh toán nếu chưa thanh toán
                                    if not record.get("paid", False):
                                        # Update revenue - tính phí theo thời gian
                                        now = datetime.datetime.now()
                                        diff = now - record["time"]
                                        mins = max(1, int(diff.total_seconds() // 60))
                                        fee = calculate_parking_fee(mins)
                                        self._total_revenue += fee

                                        # Mark as paid (GIỮ trong in_records để xe ra sau)
                                        record["paid"] = True
                                        self._paid_cards[card_id] = datetime.datetime.now()

                                        # ✅ UPDATE database - SET fee và payment_status
                                        conn = self._get_db_connection()
                                        if conn:
                                            try:
                                                cur = conn.cursor()
                                                cur.execute("""
                                                    UPDATE parking_sessions
                                                    SET fee = %s,
                                                        payment_status = 'paid',
                                                        payment_method = 'online'
                                                    WHERE session_id = %s AND time_out IS NULL
                                                """, (fee, session_id))
                                                conn.commit()
                                                logging.info(f"[DATABASE] Updated payment: session_id={session_id}, fee={fee:,}")
                                            except Exception as e:
                                                conn.rollback()
                                                logging.error(f"[DATABASE] Update payment error: {e}")
                                            finally:
                                                conn.close()

                                        logging.info(f"✅ Payment processed via terminal: Card={card_id}, Plate={plate}, Fee={fee:,}, Total={self._total_revenue:,}")
                                        logging.info(f"📝 Vehicle STILL in parking (will exit via RFID-B OUT)")

                                        # Update UI using signal (thread-safe)
                                        self.trigger_update_revenue.emit(card_id, plate, fee, self._total_revenue)
                                    else:
                                        logging.info(f"⚠️ Card {card_id} already paid, skipping revenue update")
                                else:
                                    logging.warning(f"❌ Card {card_id} not found in in_records")

                except Exception as e:
                    logging.error(f"Error in MQTT message handler: {e}")
            self.mqtt_client.on_connect = _on_connect
            self.mqtt_client.on_disconnect = _on_disconnect
            self.mqtt_client.on_message = _on_message
            self._set_mqtt_state("Đang kết nối…", "#f1c40f")
            self.mqtt_client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=20)
            self.mqtt_client.loop_start()
            logging.info("MQTT connection initiated")

            # QR Payment TFT Handler - COMPLETELY REMOVED
        except Exception as e:
            self._mqtt_connected = False
            self._esp_online = False
            self._set_mqtt_state(f"Lỗi MQTT: {e}", "#ff6b6b")
            logging.error(f"MQTT init error: {e}", exc_info=True)

    def restart_mqtt(self):
        logging.info("Restarting MQTT...")
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception as e:
            logging.error(f"Error stopping MQTT: {e}")
        self.mqtt_client = None
        self._mqtt_connected = False
        # Đánh dấu tất cả thiết bị offline
        for mac in self._esp_devices:
            self._esp_devices[mac]["online"] = False
        self._refresh_conn_badge()
        self._update_esp_devices_display()
        self.ensure_broker_running()
        self.init_mqtt()

    def closeEvent(self, e):
        """✅ FIXED: Proper cleanup sequence with logout protection"""
        # ✅ Chặn nút X - chỉ cho phép đóng qua nút Đăng xuất
        if not getattr(self, 'allow_close', False):
            e.ignore()  # Chặn đóng cửa sổ
            QMessageBox.warning(
                self,
                "Không thể đóng",
                "Vui lòng sử dụng nút 'Đăng xuất' để thoát ứng dụng!"
            )
            logging.warning("[CLOSE BLOCKED] User tried to close without logout")
            return

        logging.info("=== CLOSING APPLICATION ===")

        # 0. Stop payment polling
        logging.info("Step 0: Stopping payment polling...")
        self._stop_payment_polling()

        # 1. No need to save data - already in PostgreSQL database
        logging.info("Step 1: Data already in database, skipping save...")

        # 2. Stop cameras
        logging.info("Step 2: Stopping cameras...")
        self.stop_cameras()
        QThread.msleep(500)

        # 3. Stop MQTT
        logging.info("Step 3: Stopping MQTT...")
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception as e:
            logging.error(f"Error stopping MQTT: {e}")
        
        # 4. Cleanup ALPR resources
        logging.info("Step 4: Cleaning up ALPR...")
        try:
            if self.alpr:
                self.alpr.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up ALPR: {e}")

        # 5. Stop Mosquitto broker
        logging.info("Step 5: Stopping broker...")
        if self._mosq_proc:
            try:
                self._mosq_proc.terminate()
                self._mosq_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._mosq_proc.kill()
                self._mosq_proc.wait()
            except Exception as e:
                logging.error(f"Error stopping broker: {e}")
        
        logging.info("=== APPLICATION CLOSED ===")
        super().closeEvent(e)

    def _on_qr_payment_cancelled(self):
        """Xử lý thanh toán QR bị hủy"""
        try:
            if not hasattr(self, 'qr_dialog') or not self.qr_dialog.payment_info:
                return

            payment_info = self.qr_dialog.payment_info
            session_id = payment_info.get('session_id')

            logging.warning(f"❌ QR Payment CANCELLED for session {session_id}")

            # ✅ STOP POLLING KHI HỦY
            self._stop_payment_polling()

            # Show cancellation message
            QMessageBox.warning(self, "Thanh toán bị hủy",
                              f"Thanh toán cho biển số {payment_info.get('plate_number')} đã bị hủy.")

        except Exception as e:
            logging.error(f"Error in _on_qr_payment_cancelled: {e}")

# =================================================================================================
# LOGIN DIALOG
# =================================================================================================

class LoginDialog(QDialog):
    """Dialog đăng nhập cho nhân viên"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đăng nhập - Hệ thống quản lý bãi đỗ xe")
        self.setModal(True)
        self.setFixedSize(400, 250)

        self.user_info = None  # Lưu thông tin user sau khi login thành công

        # Import password checking
        try:
            from werkzeug.security import check_password_hash
            self.check_password_hash = check_password_hash
        except ImportError:
            QMessageBox.critical(self, "Lỗi", "Thiếu module werkzeug. Cài đặt: pip install werkzeug")
            sys.exit(1)

        # Layout
        layout = QVBoxLayout()

        # Logo/Title
        title = QLabel("HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #007bff; margin: 10px;")
        layout.addWidget(title)

        # Form
        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Nhập username...")
        self.username_input.setMinimumHeight(35)
        form_layout.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Nhập mật khẩu...")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(35)
        self.password_input.returnPressed.connect(self.do_login)  # Enter để login
        form_layout.addRow("Mật khẩu:", self.password_input)

        layout.addLayout(form_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.login_btn = QPushButton("Đăng nhập")
        self.login_btn.setMinimumSize(100, 35)
        self.login_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.login_btn.clicked.connect(self.do_login)
        btn_layout.addWidget(self.login_btn)

        self.cancel_btn = QPushButton("Thoát")
        self.cancel_btn.setMinimumSize(100, 35)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        self.setLayout(layout)

        # Focus vào username
        self.username_input.setFocus()

    def do_login(self):
        """Xử lý đăng nhập"""
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đầy đủ thông tin!")
            return

        try:
            # Kết nối database với timezone VN (dùng config)
            cfg = load_config()
            conn = psycopg2.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                database=cfg.db_name,
                user=cfg.db_user,
                password=cfg.db_password,
                options="-c timezone=Asia/Ho_Chi_Minh"  # ✅ Set timezone khi connect
            )
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Lấy thông tin user
            cur.execute("""
                SELECT user_id, username, password_hash, full_name, role, is_active
                FROM users
                WHERE username = %s
            """, (username,))

            user = cur.fetchone()

            if not user:
                QMessageBox.warning(self, "Đăng nhập thất bại", "Tên đăng nhập không tồn tại!")
                self.password_input.clear()
                return

            if not user['is_active']:
                QMessageBox.warning(self, "Đăng nhập thất bại", "Tài khoản đã bị khóa!")
                return

            # Kiểm tra mật khẩu
            if not self.check_password_hash(user['password_hash'], password):
                QMessageBox.warning(self, "Đăng nhập thất bại", "Mật khẩu không đúng!")
                self.password_input.clear()
                return

            # Cập nhật last_login
            cur.execute("""
                UPDATE users
                SET last_login = NOW()
                WHERE user_id = %s
            """, (user['user_id'],))
            conn.commit()

            # Lưu thông tin user
            self.user_info = {
                'user_id': user['user_id'],
                'username': user['username'],
                'full_name': user['full_name'],
                'role': user['role']
            }

            cur.close()
            conn.close()

            # Đóng dialog và trả về success
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi kết nối database:\n{str(e)}")
            logging.error(f"Login error: {e}")

# =================================================================================================
# BOOT
# =================================================================================================

def main():
    logging.info("=" * 80)
    logging.info("Starting Parking Management Application")
    logging.info("=" * 80)

    app = QApplication(sys.argv)

    # ✅ Clear Qt settings để reset window state
    app.setOrganizationName("ParkingSystem")
    app.setApplicationName("ParkingUI")

    # ✅ Hiển thị dialog đăng nhập
    login_dialog = LoginDialog()
    if login_dialog.exec() != QDialog.Accepted:
        logging.info("Login cancelled - Application closed")
        return  # Người dùng hủy đăng nhập hoặc đóng dialog

    # Lấy thông tin user đã đăng nhập
    user_info = login_dialog.user_info
    if not user_info:
        logging.error("Login failed - No user info")
        return

    logging.info(f"User logged in: {user_info['username']} ({user_info['role']})")

    # Load config và khởi tạo app
    cfg = load_config()
    if not os.path.exists(cfg.broker_conf):
        Path(cfg.broker_conf).parent.mkdir(parents=True, exist_ok=True)
        open(cfg.broker_conf, "w", encoding="utf-8").write(
            "listener 1883 0.0.0.0\nallow_anonymous true\npersistence false\n"
        )

    # Tạo MainWindow và truyền thông tin user
    w = MainWindow(cfg)
    w.current_user = user_info  # Lưu thông tin user vào MainWindow

    # ✅ Tạo staff session trong database (dùng server time để tránh lệch)
    try:
        import socket
        from datetime import datetime
        conn = psycopg2.connect(
            host=cfg.db_host,
            port=cfg.db_port,
            database=cfg.db_name,
            user=cfg.db_user,
            password=cfg.db_password,
            options="-c timezone=Asia/Ho_Chi_Minh"  # ✅ Set timezone VN khi connect
        )
        cur = conn.cursor()

        device_name = socket.gethostname()
        ip_address = socket.gethostbyname(device_name)

        # ✅ Dùng NOW() của server với timezone VN
        cur.execute("""
            INSERT INTO staff_sessions (user_id, is_active, device_name, ip_address)
            VALUES (%s, true, %s, %s)
            RETURNING session_id, login_time
        """, (user_info['user_id'], device_name, ip_address))

        result = cur.fetchone()
        w.staff_session_id = result[0]
        w.login_time = result[1]  # ✅ Lấy thời gian từ server (chính xác)

        conn.commit()
        cur.close()
        conn.close()

        logging.info(f"[SESSION] Created staff session ID: {w.staff_session_id}, login_time: {w.login_time}")
    except Exception as e:
        logging.error(f"[SESSION] Failed to create staff session: {e}")
        w.staff_session_id = None
        w.login_time = datetime.now()  # Fallback
    # ✅ Cập nhật thông tin nhân viên trong status bar
    staff_text = f"👤 {user_info['full_name'] or user_info['username']} ({user_info['role'].upper()}) - Đăng nhập: {w.login_time.strftime('%H:%M:%S')}"
    w.lbl_staff_info.setText(staff_text)

    # ✅ Hiển thị FULLSCREEN
    w.showFullScreen()

    # Hiển thị thông tin nhân viên trong title bar
    w.setWindowTitle(f"Parking Management - {user_info['full_name'] or user_info['username']} ({user_info['role'].upper()})")

    logging.info(f"Application shown FULLSCREEN - User: {user_info['username']}")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()