import os
import sys
import threading
import hashlib
import json
import re
import time
import ctypes
import logging
from pathlib import Path

# Настройка DPI Awareness для корректного масштабирования интерфейса в Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Отключение специфичных флагов PaddlePaddle для стабильности инференса
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_cuda_graph'] = '0'

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import mss
from pynput import keyboard
from dotenv import load_dotenv

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    logging.warning(f"PaddleOCR недоступен: {e}")
    PADDLE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()

APP_DIR = Path(__file__).resolve().parent

# Базовая настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


class Config:
    API_KEY = os.getenv("API_KEY", "")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    API_MODEL = os.getenv("API_MODEL", "openai/gpt-oss-120b")
    
    PROVIDER_PRESETS = {
        "Groq": {"base_url": "https://api.groq.com/openai/v1", "description": "Groq API"},
        "OpenAI": {"base_url": "https://api.openai.com/v1", "description": "OpenAI API"},
        "LocalAI": {"base_url": "http://localhost:8080/v1", "description": "LocalAI"},
        "Ollama": {"base_url": "http://localhost:11434/v1", "description": "Ollama"},
        "Custom": {"base_url": "", "description": "Custom URL"}
    }
    
    GROQ_API_KEY = API_KEY
    GROQ_MODEL = API_MODEL
    
    HOTKEY_TRANSLATE = os.getenv("HOTKEY_TRANSLATE", "q")
    FONT_PATH = os.getenv("FONT", "arial.ttf")

    BUBBLE_MODE = os.getenv("BUBBLE_MODE", "manga")
    DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True")    
    
    OCR_LANGS = ['en']
    FONT_SIZE = 24
    MIN_FONT_SIZE = 14
    MAX_FONT_SIZE = 40
    FONT_SIZE_MULTIPLIER = 1.0
    TEXT_COLOR = (0, 0, 0)
    
    USE_WHITE_BOX = True
    USE_CACHE = True
    CACHE_DIR = ".cache"
    BUBBLE_PADDING = 0.15
    
    PADDLE_DET_THRESH = 0.15
    PADDLE_BOX_THRESH = 0.35
    PADDLE_UNCLIP_RATIO = 1.8
    
    MIN_CONFIDENCE = 0.3
    MIN_TEXT_LENGTH = 2
    
    UI_BG_MAIN = "#0a0a0a"
    UI_BG_SIDEBAR = "#141414"
    UI_BG_BUTTON = "#1f1f1f"
    UI_BG_BUTTON_ACTIVE = "#333333"
    UI_TEXT_MAIN = "#e0e0e0"
    UI_TEXT_DIM = "#888888"
    UI_ACCENT = "#00ff9d"
    UI_BORDER = "#00ff9d"
    
    BUBBLE_MODE = 'manga'
    
    MANGA_SETTINGS = {
        'bubble_shape_preference': 'round',
        'text_alignment': 'center',
        'margin_multiplier': 0.15,
        'font_size_multiplier': 0.9,
    }
    
    MANHWA_SETTINGS = {
        'bubble_shape_preference': 'mixed',
        'text_alignment': 'left',
        'margin_multiplier': 0.1,
        'font_size_multiplier': 1.0,
    }
    MANHWA_OUTSIDE_MIN_WORDS = 3
    
    @classmethod
    def get_current_settings(cls):
        return cls.MANGA_SETTINGS if cls.BUBBLE_MODE == 'manga' else cls.MANHWA_SETTINGS


class EnhancedOCRService:
    def __init__(self, langs=None):
        self.ocr = None
        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(
                lang=(langs or ['en'])[0], 
                use_angle_cls=True, 
                use_gpu=False, 
                det_db_thresh=0.3,
                det_db_unclip_ratio=2.0,
                show_log=False
            )
    
    def detect_with_separation(self, img, bubbles):
        """OCR с принудительным разделением близко расположенных облаков."""
        results_per_bubble = []
        
        for bx1, by1, bx2, by2 in bubbles:
            margin_x = int((bx2 - bx1) * 0.12)
            margin_y = int((by2 - by1) * 0.12)
            
            bx1_c = max(0, bx1 + margin_x)
            by1_c = max(0, by1 + margin_y)
            bx2_c = min(img.shape[1], bx2 - margin_x)
            by2_c = min(img.shape[0], by2 - margin_y)
            
            if bx2_c - bx1_c < 20 or by2_c - by1_c < 20:
                results_per_bubble.append([])
                continue
            
            bubble_roi = img[by1_c:by2_c, bx1_c:bx2_c]
            gray = cv2.cvtColor(bubble_roi, cv2.COLOR_RGB2GRAY) if len(bubble_roi.shape) == 3 else bubble_roi.copy()
            
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
            processed = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB) if num_labels > 10 else cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            ocr_result = self.ocr.ocr(processed, cls=True)
            
            bubble_texts = []
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if len(line) >= 2:
                        box = line[0]
                        txt = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                        
                        if txt and txt.strip():
                            adjusted_box = [[p[0] + bx1_c, p[1] + by1_c] for p in box]
                            bubble_texts.append((adjusted_box, txt.strip(), float(conf)))
            
            results_per_bubble.append(bubble_texts)
        
        return results_per_bubble


class SimpleCache:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
    def _get_key(self, *args):
        data = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(data.encode()).hexdigest()
        
    def get(self, *args):
        if not Config.USE_CACHE: 
            return None
        path = os.path.join(self.cache_dir, f"{self._get_key(*args)}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: 
                    return json.load(f)
            except Exception: 
                pass
        return None
        
    def set(self, value, *args):
        if not Config.USE_CACHE: 
            return
        path = os.path.join(self.cache_dir, f"{self._get_key(*args)}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f: 
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception: 
            pass


def clean_text(text):
    if not text: 
        return ""
    return '\n'.join(re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n') if line.strip())


def split_merged(text):
    if not text: 
        return ""
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


def postprocess(text):
    if not text: 
        return ""
    return '\n'.join(re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n') if line.strip())


def check_overlap(b1, b2, th=10):
    x1 = max(min(p[0] for p in b1), min(p[0] for p in b2))
    y1 = max(min(p[1] for p in b1), min(p[1] for p in b2))
    x2 = min(max(p[0] for p in b1), max(p[0] for p in b2))
    y2 = min(max(p[1] for p in b1), max(p[1] for p in b2))
    return (x2 - x1 > th) and (y2 - y1 > th)


def deduplicate(res, iou=0.7):
    if len(res) <= 1: 
        return res
    kept = []
    for bbox, text, conf in sorted(res, key=lambda x: x[2], reverse=True):
        is_dup = False
        for k_bbox, k_text, k_conf in kept:
            ix1 = max(min(p[0] for p in bbox), min(p[0] for p in k_bbox))
            iy1 = max(min(p[1] for p in bbox), min(p[1] for p in k_bbox))
            ix2 = min(max(p[0] for p in bbox), max(p[0] for p in k_bbox))
            iy2 = min(max(p[1] for p in bbox), max(p[1] for p in k_bbox))
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area1 = (max(p[0] for p in bbox)-min(p[0] for p in bbox))*(max(p[1] for p in bbox)-min(p[1] for p in bbox))
            area2 = (max(p[0] for p in k_bbox)-min(p[0] for p in k_bbox))*(max(p[1] for p in k_bbox)-min(p[1] for p in k_bbox))
            union = area1 + area2 - inter
            if union > 0 and inter/union > iou:
                is_dup = True
                if conf > k_conf:
                    kept.remove((k_bbox, k_text, k_conf))
                    kept.append((bbox, text, conf))
                break
        if not is_dup: 
            kept.append((bbox, text, conf))
    return kept


def filter_noise(res):
    out = []
    valid_symbols = {'...', '....', '.....', '!', '!!', '!!!', '?', '??', '???', '—', '–'}
    for bbox, text, conf in res:
        text = text.strip()
        if len(text) < 2:
            if text in valid_symbols:
                out.append((bbox, text, conf))
            continue
        
        if conf < Config.MIN_CONFIDENCE: 
            continue
        
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.1:
            continue
            
        out.append((bbox, text, conf))
    return out


def get_bbox_y(bbox):
    return min(p[1] for p in bbox)


def get_bbox_rect(bbox):
    return (
        min(p[0] for p in bbox), min(p[1] for p in bbox),
        max(p[0] for p in bbox), max(p[1] for p in bbox)
    )


def rect_intersection_area(r1, r2):
    ax1, ay1, ax2, ay2 = r1
    bx1, by1, bx2, by2 = r2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def assign_ocr_to_bubbles(ocr_results, bubbles):
    """Назначает OCR-блоки ближайшим облакам на основе пересечения и расстояния."""
    assignments = {i: [] for i in range(len(bubbles))}
    if not ocr_results or not bubbles:
        return assignments

    bubble_rects = [tuple(map(float, b)) for b in bubbles]

    for item in ocr_results:
        bbox, txt, conf = item
        tx1, ty1, tx2, ty2 = map(float, get_bbox_rect(bbox))
        text_rect = (tx1, ty1, tx2, ty2)
        text_cx, text_cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0

        best_idx, best_overlap, best_dist = None, -1.0, float('inf')

        for i, (bx1, by1, bx2, by2) in enumerate(bubble_rects):
            expanded = (bx1 - 6.0, by1 - 6.0, bx2 + 6.0, by2 + 6.0)
            overlap = rect_intersection_area(text_rect, expanded)
            if overlap <= 0:
                continue

            bubble_cx, bubble_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = ((text_cx - bubble_cx) ** 2 + (text_cy - bubble_cy) ** 2) ** 0.5

            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best_idx, best_overlap, best_dist = i, overlap, dist

        if best_idx is not None:
            assignments[best_idx].append(item)

    return assignments


def compute_overlap_safe_text_boxes(bubbles, base_margin=10):
    """Строит безопасные bounding box'ы для текста с учетом пересечений облаков."""
    n = len(bubbles)
    if n == 0:
        return [], []

    extra_pads = [{'l': 0, 't': 0, 'r': 0, 'b': 0} for _ in range(n)]

    for i in range(n):
        x1, y1, x2, y2 = bubbles[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        for j in range(n):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = bubbles[j]
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter_w, inter_h = ix2 - ix1, iy2 - iy1
            inter_cx, inter_cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0

            if inter_w >= inter_h:
                pad = int(inter_h * 0.6) + 6
                if inter_cy <= cy:
                    extra_pads[i]['t'] = max(extra_pads[i]['t'], pad)
                else:
                    extra_pads[i]['b'] = max(extra_pads[i]['b'], pad)
            else:
                pad = int(inter_w * 0.6) + 6
                if inter_cx <= cx:
                    extra_pads[i]['l'] = max(extra_pads[i]['l'], pad)
                else:
                    extra_pads[i]['r'] = max(extra_pads[i]['r'], pad)

    safe_boxes, collision_scores = [], []
    for i, (x1, y1, x2, y2) in enumerate(bubbles):
        w, h = x2 - x1, y2 - y1
        extra_total = sum(extra_pads[i].values())
        collision_scores.append(extra_total)

        left = x1 + base_margin + extra_pads[i]['l']
        top = y1 + base_margin + extra_pads[i]['t']
        right = x2 - base_margin - extra_pads[i]['r']
        bottom = y2 - base_margin - extra_pads[i]['b']

        min_w, min_h = max(24, int(w * 0.42)), max(20, int(h * 0.42))
        if (right - left) < min_w or (bottom - top) < min_h:
            left = x1 + base_margin + int(extra_pads[i]['l'] * 0.45)
            top = y1 + base_margin + int(extra_pads[i]['t'] * 0.45)
            right = x2 - base_margin - int(extra_pads[i]['r'] * 0.45)
            bottom = y2 - base_margin - int(extra_pads[i]['b'] * 0.45)

        if right - left < 12:
            cx = (x1 + x2) // 2
            left, right = cx - 6, cx + 6
        if bottom - top < 12:
            cy = (y1 + y2) // 2
            top, bottom = cy - 6, cy + 6

        safe_boxes.append((left, top, right, bottom))

    return safe_boxes, collision_scores


def rects_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def shrink_rect(rect, ratio=0.04):
    x1, y1, x2, y2 = rect
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    dx, dy = max(1, int(w * ratio)), max(1, int(h * ratio))
    nx1, ny1 = x1 + dx, y1 + dy
    nx2, ny2 = x2 - dx, y2 - dy
    if nx2 - nx1 < 14:
        cx = (x1 + x2) // 2
        nx1, nx2 = cx - 7, cx + 7
    if ny2 - ny1 < 14:
        cy = (y1 + y2) // 2
        ny1, ny2 = cy - 7, cy + 7
    return (nx1, ny1, nx2, ny2)


def normalize_overlapping_bubbles(bubbles, img_shape):
    """Сужает пузыри и разводит сильные пересечения для предотвращения склейки."""
    if not bubbles:
        return []
    h, w = img_shape[:2]
    norm = [shrink_rect(tuple(map(int, b)), ratio=0.015) for b in bubbles]

    for _ in range(2):
        changed = False
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                ax1, ay1, ax2, ay2 = norm[i]
                bx1, by1, bx2, by2 = norm[j]
                inter = rect_intersection_area(norm[i], norm[j])
                if inter <= 0:
                    continue
                area_i = max(1, (ax2 - ax1) * (ay2 - ay1))
                area_j = max(1, (bx2 - bx1) * (by2 - by1))
                overlap_ratio = inter / float(min(area_i, area_j))
                if overlap_ratio < 0.24:
                    continue

                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = ix2 - ix1, iy2 - iy1
                push = max(1, int((iw if iw > ih else ih) * 0.18))

                if iw >= ih:
                    if (ax1 + ax2) <= (bx1 + bx2):
                        ax2, bx1 = ax2 - push, bx1 + push
                    else:
                        ax1, bx2 = ax1 + push, bx2 - push
                else:
                    if (ay1 + ay2) <= (by1 + by2):
                        ay2, by1 = ay2 - push, by1 + push
                    else:
                        ay1, by2 = ay1 + push, by2 - push

                ax1, ay1 = max(0, ax1), max(0, ay1)
                bx1, by1 = max(0, bx1), max(0, by1)
                ax2, ay2 = min(w, ax2), min(h, ay2)
                bx2, by2 = min(w, bx2), min(h, by2)
                if ax2 - ax1 < 18 or ay2 - ay1 < 18 or bx2 - bx1 < 18 or by2 - by1 < 18:
                    continue

                norm[i] = (ax1, ay1, ax2, ay2)
                norm[j] = (bx1, by1, bx2, by2)
                changed = True
        if not changed:
            break

    return norm


def likely_speech_bubble_region(img, rect):
    """Проверяет наличие характерной обводки для отсеивания фонового текста."""
    x1, y1, x2, y2 = map(int, rect)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return False

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
    edges = cv2.Canny(gray, 70, 170)

    bh, bw = edges.shape[:2]
    b = max(2, int(min(bh, bw) * 0.06))
    border = np.zeros_like(edges, dtype=np.uint8)
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = 1

    border_pixels = int(np.sum(border))
    if border_pixels == 0:
        return False
    edge_ratio = float(np.sum((edges > 0) & (border > 0))) / float(border_pixels)
    return edge_ratio > 0.055


def build_outside_text_boxes(full_ocr_results, bubbles, img_shape):
    """Формирует bounding box'ы для текста вне облаков (режим manhwa)."""
    if not full_ocr_results:
        return []

    h, w = img_shape[:2]
    out = []
    bubble_rects = [tuple(map(int, b)) for b in bubbles]
    candidates = []

    for bbox, txt, conf in full_ocr_results:
        if not txt.strip():
            continue

        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", txt)
        if len(words) < Config.MANHWA_OUTSIDE_MIN_WORDS:
            continue

        tx1, ty1, tx2, ty2 = map(int, get_bbox_rect(bbox))
        cx, cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        text_rect = (tx1, ty1, tx2, ty2)
        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))

        in_bubble = False
        for bx1, by1, bx2, by2 in bubble_rects:
            expanded = (bx1 - 4, by1 - 4, bx2 + 4, by2 + 4)
            if expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]:
                in_bubble = True
                break
            inter = rect_intersection_area(text_rect, expanded)
            if inter / text_area > 0.2:
                in_bubble = True
                break
        if in_bubble:
            continue

        candidates.append({'rect': (tx1, ty1, tx2, ty2), 'text': txt.strip(), 'conf': conf})

    candidates.sort(key=lambda c: (c['rect'][1], c['rect'][0]))
    groups = []
    for c in candidates:
        cx1, cy1, cx2, cy2 = c['rect']
        cw, ch = cx2 - cx1, cy2 - cy1
        ccx = (cx1 + cx2) / 2.0

        best_g, best_score = None, -1e9
        for g in groups:
            gx1, gy1, gx2, gy2 = g['rect']
            gw, gh = gx2 - gx1, gy2 - gy1
            gcx = (gx1 + gx2) / 2.0
            v_gap = cy1 - gy2
            overlap_x = max(0, min(cx2, gx2) - max(cx1, gx1))
            overlap_ratio = overlap_x / max(1, min(cw, gw))
            center_dx = abs(ccx - gcx)
            if v_gap < -max(ch, gh) * 0.35:
                continue
            if v_gap <= max(14, int(min(ch, gh) * 1.1)) and (overlap_ratio >= 0.18 or center_dx <= max(cw, gw) * 0.55):
                score = -abs(v_gap) - center_dx * 0.02 + overlap_ratio * 10.0
                if score > best_score:
                    best_score, best_g = score, g

        if best_g is None:
            groups.append({'rect': [cx1, cy1, cx2, cy2], 'lines': [c['text']], 'conf': [c['conf']]})
        else:
            gx1, gy1, gx2, gy2 = best_g['rect']
            best_g['rect'] = [min(gx1, cx1), min(gy1, cy1), max(gx2, cx2), max(gy2, cy2)]
            best_g['lines'].append(c['text'])
            best_g['conf'].append(c['conf'])

    for g in groups:
        gx1, gy1, gx2, gy2 = map(int, g['rect'])
        gw, gh = max(1, gx2 - gx1), max(1, gy2 - gy1)
        pad_x, pad_y = max(4, int(gw * 0.07)), max(4, int(gh * 0.12))
        rx1, ry1 = max(0, gx1 - pad_x), max(0, gy1 - pad_y)
        rx2, ry2 = min(w, gx2 + pad_x), min(h, gy2 + pad_y)
        candidate = [rx1, ry1, rx2, ry2]
        merged_text = '\n'.join(g['lines'])
        merged_conf = float(sum(g['conf']) / max(1, len(g['conf'])))

        blockers = bubble_rects + [tuple(b['rect']) for b in out]
        shifted = False
        for _ in range(6):
            if not any(rects_overlap(tuple(candidate), b) for b in blockers):
                shifted = True
                break
            candidate[1], candidate[3] = min(h - 1, candidate[1] + 3), min(h, candidate[3] + 3)
        if not shifted and any(rects_overlap(tuple(candidate), b) for b in blockers):
            continue

        max_w, max_h = int(w * 0.82), int(h * 0.34)
        cw, ch = candidate[2] - candidate[0], candidate[3] - candidate[1]
        if cw > max_w:
            ccx = (candidate[0] + candidate[2]) // 2
            candidate[0] = max(0, ccx - max_w // 2)
            candidate[2] = min(w, candidate[0] + max_w)
        if ch > max_h:
            ccy = (candidate[1] + candidate[3]) // 2
            candidate[1] = max(0, ccy - max_h // 2)
            candidate[3] = min(h, candidate[1] + max_h)

        out.append({'rect': tuple(candidate), 'text': merged_text, 'conf': merged_conf})

    return out


class BubbleAnalyzer:
    def __init__(self, mode='manga'):
        self.mode = mode
        
    def analyze_shape(self, bubble_img, bbox):
        """Определяет геометрическую форму пузыря для корректного рендеринга текста."""
        h, w = bubble_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 1
        
        gray = cv2.cvtColor(bubble_img, cv2.COLOR_RGB2GRAY) if len(bubble_img.shape) == 3 else bubble_img.copy()
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'rectangular', aspect_ratio
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return 'rectangular', aspect_ratio
            
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if self.mode == 'manga':
            return ('round', aspect_ratio) if (circularity > 0.7 and 0.7 <= aspect_ratio <= 1.3) else ('rectangular', aspect_ratio)
        else:
            if (circularity > 0.62 and 0.62 <= aspect_ratio <= 1.55) or \
               (circularity > 0.55 and 0.78 <= aspect_ratio <= 1.32 and len(approx) <= 8):
                return 'round', aspect_ratio
            elif len(approx) == 4:
                return 'rectangular', aspect_ratio
            return 'irregular', aspect_ratio


def separate_adjacent_bubbles(bubbles, ocr_results, min_gap=15):
    """Распределяет распознанный текст между ближайшими облаками."""
    bubble_assignments = {i: [] for i in range(len(bubbles))}
    
    for bbox, text, conf in ocr_results:
        text_center_x = sum(p[0] for p in bbox) / 4
        text_center_y = sum(p[1] for p in bbox) / 4
        
        min_dist, closest_bubble = float('inf'), None
        
        for i, (bx1, by1, bx2, by2) in enumerate(bubbles):
            if bx1 <= text_center_x <= bx2 and by1 <= text_center_y <= by2:
                bubble_center_x = (bx1 + bx2) / 2
                bubble_center_y = (by1 + by2) / 2
                dist = ((text_center_x - bubble_center_x)**2 + (text_center_y - bubble_center_y)**2)**0.5
                
                if dist < min_dist:
                    min_dist, closest_bubble = dist, i
        
        if closest_bubble is not None:
            bubble_assignments[closest_bubble].append((bbox, text, conf))
    
    return bubble_assignments


class AdvancedPreprocessor:
    @staticmethod
    def process(img):
        if img is None or img.size == 0: 
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img.copy()
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        if np.mean(closed) < 127: 
            closed = cv2.bitwise_not(closed)
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)


class BubbleDetector:
    def __init__(self, path=None):
        if path is None:
            path = APP_DIR / "models" / "comic-speech-bubble-detector.pt"
        
        self.model = None
        self.available = False
        
        logging.info(f"Поиск модели YOLO: {path}")
        
        if YOLO_AVAILABLE and os.path.exists(path):
            try:
                self.model = YOLO(str(path))
                self.available = True
                logging.info("Модель BubbleDetector успешно загружена")
            except Exception as e:
                logging.error(f"Ошибка загрузки BubbleDetector: {e}")
        else:
            logging.warning("Модель BubbleDetector не найдена или YOLO недоступен")
        
    def detect(self, img, conf=0.15):
        if not self.available or self.model is None: 
            logging.warning("Детектор облаков недоступен")
            return []
        try:
            results = self.model(img, conf=conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            bubbles = []
            for box, conf_val in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w > 15 and h > 15:
                    bubbles.append((int(x1), int(y1), int(x2), int(y2)))
            
            return bubbles
        except Exception as e:
            logging.error(f"Ошибка детекции облаков: {e}")
            return []


class OCRService:
    def __init__(self, langs=None):
        self.ocr = None
        self.available = False
        if PADDLE_AVAILABLE:
            try:
                self.ocr = PaddleOCR(
                    lang=(langs or ['en'])[0], 
                    use_angle_cls=True, 
                    use_gpu=False, 
                    det_db_thresh=Config.PADDLE_DET_THRESH, 
                    show_log=False
                )
                self.available = True
                logging.info("PaddleOCR успешно инициализирован")
            except Exception as e:
                logging.error(f"Ошибка инициализации PaddleOCR: {e}")
            
    def detect(self, img):
        if not self.available or self.ocr is None: 
            return []
        try:
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            result = self.ocr.ocr(img_rgb, cls=True)
            
            if not result or not result[0]:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )
                processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                result = self.ocr.ocr(processed, cls=True)

            out = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        box = line[0]
                        res_data = line[1]
                        txt = res_data[0] if isinstance(res_data, (list, tuple)) else str(res_data)
                        conf = res_data[1] if isinstance(res_data, (list, tuple)) and len(res_data)>1 else 1.0
                        
                        if txt and txt.strip(): 
                            out.append((box, txt.strip(), float(conf)))
            return out
        except Exception as e:
            logging.error(f"Ошибка OCR: {e}")
            return []


class TranslationService:
    def __init__(self, key, base_url, model):
        self.client = OpenAI(base_url=base_url, api_key=key) if key and OPENAI_AVAILABLE else None
        self.base_url = base_url
        self.model = model
        self.cache = SimpleCache(os.path.join(Config.CACHE_DIR, "trans"))
        
    def translate(self, texts):
        if not texts or not self.client: 
            return texts
        cached = self.cache.get(texts, self.model)
        if cached: 
            return cached
        
        SEP = "|||SEP|||"
        n_bubbles = len(texts)
        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_idx:
            return [""] * n_bubbles
        packed_texts = [texts[i] for i in non_empty_idx]
        input_str = SEP.join(packed_texts)
        
        system_prompt = """Ты — профессиональный редактор и переводчик манги (EN -> RU).

        ТВОИ ЗАДАЧИ:
        1. ИСПРАВЛЯЙ ошибки OCR:
           - Разделяй слитные слова: "ТЫВСЁЕЩЁДОВЕРЯЕШЬ" -> "ТЫ ВСЁ ЕЩЁ ДОВЕРЯЕШЬ"
           - Восстанавливай пропущенные пробелы и знаки препинания
           - Исправляй опечатки: "ПОЛОВИНУНЕ" -> "ПОЛОВИНУ НЕ"
        
        2. Переводи с английского на русский:
           - Сохраняй смысл и тон оригинала
           - Используй грамотный русский язык
           - Перевод должен звучать естественно, а не как дословная калька
        
        3. СТРОГИЕ ПРАВИЛА:
           - ВХОД: {n} облаков -> ВЫХОД: РОВНО {n} переводов
           - В ПЕРЕВОДЕ НЕ ДОЛЖНО БЫТЬ АНГЛИЙСКИХ СЛОВ
           - НЕ объединяй облака и НЕ разбивай одно на несколько
           - НЕ додумывай продолжение текста
           - НЕ бери текст из соседних облаков
           - Каждое облако переводится ИЗОЛИРОВАННО
           - НЕ СОКРАЩАЙ и НЕ упрощай реплики
        
        4. ФОРМАТ:
           - Разделяй переводы символом |||SEP|||
           - Без кавычек, вступлений и пояснений
           - Если вход пуст -> верни пустую строку
        """
        
        user_prompt = f"Облаков: {len(packed_texts)}\nВход (может содержать ошибки OCR):\n{repr(input_str)}\nПеревод с исправлением ошибок:"
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                top_p=0.5,
                max_tokens=1500,
                stop=["Примечание:", "Вот перевод:", "P.S."]
            )
            raw = resp.choices[0].message.content.strip()
            
            for junk in ["Вот перевод:", "Перевод:", "Ответ:", "Исправлено:"]:
                raw = raw.replace(junk, "").strip()
                
            parts = raw.split(SEP)
            packed_out = [p.strip().strip('"\'') for p in parts]
            while len(packed_out) < len(packed_texts):
                packed_out.append("")
            packed_out = packed_out[:len(packed_texts)]

            out = [""] * n_bubbles
            for k, src_i in enumerate(non_empty_idx):
                out[src_i] = packed_out[k] if k < len(packed_out) else ""
            
            self.cache.set(out, texts, self.model)
            return out
            
        except Exception as e:
            logging.error(f"Ошибка перевода: {e}")
            return texts

    def _retry_translate_single(self, text):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Translate to Russian. One word or short phrase. NO quotes."},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                temperature=0.0,
                max_tokens=100
            )
            return resp.choices[0].message.content.strip().strip("'\"")
        except Exception:
            return text


class FontManager:
    def __init__(self, path, size=24):
        self.path, self.size, self._cache = path, size, {}
        
    def get(self, size=None):
        s = size or self.size
        if s not in self._cache:
            try: 
                self._cache[s] = ImageFont.truetype(self.path, s)
            except Exception: 
                self._cache[s] = ImageFont.load_default()
        return self._cache[s]


def wrap_text(draw, text, font, max_width):
    words = text.replace('\\n', '\n').split()
    if not words: 
        return []
    lines, current_line, current_width = [], [], 0
    for word in words:
        if '\n' in word:
            if current_line: 
                lines.append(' '.join(current_line))
            current_line, current_width = [], 0
            parts = word.split('\n')
            for i, part in enumerate(parts):
                if i < len(parts) - 1 and part: 
                    lines.append(part)
                elif part:
                    current_line = [part]
                    try: 
                        current_width = draw.textbbox((0, 0), part, font=font)[2]
                    except Exception: 
                        current_width = 0
            continue
        try: 
            word_width = draw.textbbox((0, 0), word, font=font)[2]
        except Exception: 
            word_width = 50
        if current_width + word_width + (len(current_line) * 5) > max_width and current_line:
            lines.append(' '.join(current_line))
            current_line, current_width = [word], word_width
        else:
            current_line.append(word)
            current_width += word_width + 5
    if current_line: 
        lines.append(' '.join(current_line))
    return lines if lines else [text]


def render_text_adaptive(img, text, bbox, bubble_shape='round', min_font_size=None):
    """Рендерит текст внутри облака с учетом его геометрии и ограничений по размеру."""
    if not text.strip():
        return img
    
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    if w <= 0 or h <= 0:
        return np.array(pil)
    
    if bubble_shape == 'round':
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        effective_width = w * 0.7
    else:
        margin_x, margin_y = int(w * 0.08), int(h * 0.08)
        effective_width = w * 0.85
    
    base_font_size = int(min(h, w) * (0.35 if bubble_shape == 'round' else 0.4))
    min_size = Config.MIN_FONT_SIZE if min_font_size is None else max(8, min_font_size)
    font_size = max(min_size, min(Config.MAX_FONT_SIZE, base_font_size))
    
    def get_text_dimensions(text, font_size):
        font = FontManager(Config.FONT_PATH).get(font_size)
        lines = []
        
        for paragraph in text.split('\n'):
            if paragraph.strip():
                wrapped = wrap_text(draw, paragraph.strip(), font, int(effective_width) - 10)
                lines.extend(wrapped)
            else:
                lines.append('')
        
        if not lines:
            return [], 0, 0, None, 0
        
        bbox_sample = draw.textbbox((0, 0), "Hg", font=font)
        line_height = (bbox_sample[3] - bbox_sample[1]) + 3
        
        total_height = len(lines) * line_height
        max_width = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines if line), default=0)
        
        return lines, total_height, line_height, font, max_width
    
    lines, total_h, lh, font, max_w = get_text_dimensions(text, font_size)
    
    while (total_h > h - 2*margin_y or max_w > effective_width - 10) and font_size > min_size:
        font_size -= 1
        lines, total_h, lh, font, max_w = get_text_dimensions(text, font_size)
    
    start_y = y1 + (h - total_h) // 2 if bubble_shape == 'round' else y1 + margin_y + (h - 2*margin_y - total_h) // 2
    
    cy = start_y
    for line in lines:
        if not line.strip():
            cy += lh // 2
            continue
        
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]
        
        line_x = x1 + (w - line_w) // 2 if bubble_shape == 'round' else x1 + margin_x
        
        draw.text((line_x, cy), line, fill=Config.TEXT_COLOR, font=font)
        cy += lh
    
    return np.array(pil)


def fill_bubbles_adaptive(img, bubbles, mode='manga', forced_shapes=None, analysis_img=None):
    """Заполняет облака белым цветом и обводит черным контуром."""
    if not bubbles: 
        return img.copy()
    res = img.copy()
    if len(res.shape) == 2: 
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    elif res.shape[2] == 4: 
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
    src_for_analysis = analysis_img if analysis_img is not None else res

    logging.info(f"Анализ {len(bubbles)} облаков (режим: {mode})...")
    
    for idx, (x1, y1, x2, y2) in enumerate(bubbles):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        corner_size_x = max(10, int(w * 0.20))
        corner_size_y = max(10, int(h * 0.20))
        
        bright_corners = 0
        try:
            corners = [
                src_for_analysis[y1:y1+corner_size_y, x1:x1+corner_size_x],
                src_for_analysis[y1:y1+corner_size_y, x2-corner_size_x:x2],
                src_for_analysis[y2-corner_size_y:y2, x1:x1+corner_size_x],
                src_for_analysis[y2-corner_size_y:y2, x2-corner_size_x:x2]
            ]
            for corner in corners:
                if corner.size > 0 and np.mean(corner) > 200:
                    bright_corners += 1
        except Exception: 
            pass
        
        aspect_ratio = w / h if h > 0 else 1
        circularity = 0.0
        contour_vertices = 0
        try:
            bubble_roi = src_for_analysis[
                max(0, y1):min(src_for_analysis.shape[0], y2),
                max(0, x1):min(src_for_analysis.shape[1], x2)
            ]
            if bubble_roi.size > 0:
                gray = cv2.cvtColor(bubble_roi, cv2.COLOR_RGB2GRAY) if len(bubble_roi.shape) == 3 else bubble_roi
                _, binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    perimeter = cv2.arcLength(largest, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    epsilon = 0.02 * perimeter if perimeter > 0 else 0
                    approx = cv2.approxPolyDP(largest, epsilon, True) if epsilon > 0 else []
                    contour_vertices = len(approx)
        except Exception:
            pass
        
        forced_shape = forced_shapes[idx] if forced_shapes and idx < len(forced_shapes) else None

        if forced_shape == 'round':
            is_round = True
        elif forced_shape == 'rectangular':
            is_round = False
        else:
            if mode == 'manga':
                is_round = (bright_corners >= 1) or (0.3 <= aspect_ratio <= 1.0)
            else:
                is_round = (
                    (circularity > 0.58 and 0.58 <= aspect_ratio <= 1.75) or
                    (circularity > 0.5 and contour_vertices > 4 and contour_vertices <= 10 and 0.7 <= aspect_ratio <= 1.45) or
                    (bright_corners >= 2 and 0.62 <= aspect_ratio <= 1.55) or
                    (bright_corners >= 1 and circularity > 0.52 and 0.7 <= aspect_ratio <= 1.5)
                )
        
        shape_str = "КРУГЛЫЙ" if is_round else "КВАДРАТНЫЙ"
        logging.debug(f"Облако {idx} [{w}x{h} AR={aspect_ratio:.2f}]: углы={bright_corners}/4 circ={circularity:.2f} v={contour_vertices} -> {shape_str}")
        
        if is_round:
            mask = np.zeros(res.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, ((x1+x2)//2, (y1+y2)//2), (w//2, h//2), 0, 0, 360, 255, -1)
            res[mask == 255] = [255, 255, 255]
            cv2.ellipse(res, ((x1+x2)//2, (y1+y2)//2), (w//2, h//2), 0, 0, 360, (0, 0, 0), 2)
        else:
            res[y1:y2, x1:x2] = [255, 255, 255]
            cv2.rectangle(res, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
    return res


class AreaSelector:
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        self._build()
        
    def _build(self):
        with mss.mss() as sct: 
            virtual = sct.monitors[0]
        self.win = tk.Toplevel(self.parent)
        self.win.geometry(f"{virtual['width']}x{virtual['height']}+{virtual['left']}+{virtual['top']}")
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.2)
        self.win.configure(bg="black")
        self.win.config(cursor="cross")
        self.win.overrideredirect(True)
        
        self.cvs = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.cvs.place(x=0, y=0, width=virtual['width'], height=virtual['height'])
        
        tk.Label(
            self.win, text="ВЫДЕЛИТЕ ОБЛАСТЬ | ESC - ОТМЕНА", 
            bg="black", fg=Config.UI_ACCENT, font=("Consolas", 14)
        ).place(relx=0.5, rely=0.05, anchor="n")
        
        self.sx = self.sy = None
        self.rect = None
        self.cvs.bind("<ButtonPress-1>", self._press)
        self.cvs.bind("<B1-Motion>", self._drag)
        self.cvs.bind("<ButtonRelease-1>", self._release)
        self.win.bind("<Escape>", lambda e: self._cancel())
        self.win.focus_set()
        
    def _press(self, e): 
        self.sx, self.sy = e.x, e.y
        
    def _drag(self, e):
        if self.rect: 
            self.cvs.delete(self.rect)
        self.rect = self.cvs.create_rectangle(self.sx, self.sy, e.x, e.y, outline=Config.UI_BORDER, width=2, dash=(4,2))
        
    def _release(self, e):
        x1, y1 = min(self.sx, e.x), min(self.sy, e.y)
        x2, y2 = max(self.sx, e.x), max(self.sy, e.y)
        if x2 - x1 > 30 and y2 - y1 > 30:
            with mss.mss() as sct:
                virtual = sct.monitors[0]
                abs_x = virtual['left'] + x1
                abs_y = virtual['top'] + y1
                self.result = (abs_x, abs_y, x2 - x1, y2 - y1)
        self.win.destroy()
        
    def _cancel(self): 
        self.result = None
        self.win.destroy()
        
    def get(self): 
        self.win.wait_window()
        return self.result


class AreaBorderOverlay:
    def __init__(self, root): 
        self.root = root
        self.win = None
        self.canvas = None
        
    def show(self, area):
        if self.win and self.win.winfo_exists(): 
            self.win.destroy()
        if not area: 
            return
            
        ax, ay, aw, ah = area
        with mss.mss() as sct:
            monitors = sct.monitors
            target_mon = monitors[1]
            center_x, center_y = ax + aw // 2, ay + ah // 2
            
            for i in range(1, len(monitors)):
                m = monitors[i]
                if (m['left'] <= center_x < m['left'] + m['width'] and 
                    m['top'] <= center_y < m['top'] + m['height']):
                    target_mon = m
                    break
            
            self.win = tk.Toplevel(self.root)
            self.win.overrideredirect(True)
            self.win.attributes('-topmost', True)
            self.win.attributes('-transparentcolor', 'gray99')
            self.win.configure(bg='gray99')
            self.win.geometry(f"{target_mon['width']}x{target_mon['height']}+{target_mon['left']}+{target_mon['top']}")
            
            self.canvas = tk.Canvas(self.win, bg='gray99', highlightthickness=0)
            self.canvas.place(x=0, y=0, width=target_mon['width'], height=target_mon['height'])
            
            rel_x, rel_y = ax - target_mon['left'], ay - target_mon['top']
            self.canvas.create_rectangle(rel_x, rel_y, rel_x+aw, rel_y+ah, outline=Config.UI_BORDER, width=3, dash=(5, 3))
            self.canvas.create_text(rel_x+10, rel_y-15, anchor='sw', text=f"{aw}x{ah}", fill=Config.UI_ACCENT, font=("Consolas", 10, "bold"))
            self.win.update()
            
    def hide(self):
        if self.win and self.win.winfo_exists(): 
            self.win.destroy()
            self.win = None
            self.canvas = None


class ScreenOverlay:
    def __init__(self, root):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.attributes('-transparentcolor', 'gray99')
        self.win.configure(bg='gray99')
        self.win.withdraw()
        
        self.canvas = tk.Canvas(self.win, bg='gray99', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', lambda e: self.hide())
        self.win.bind('<Key-q>', lambda e: self.hide())
        self.win.bind('<Key-Q>', lambda e: self.hide())
        
    def show(self, area):
        if not area: 
            return
        ax, ay, aw, ah = area
        
        with mss.mss() as sct:
            monitors = sct.monitors
            target_mon = monitors[1]
            center_x, center_y = ax + aw // 2, ay + ah // 2
            
            for i in range(1, len(monitors)):
                m = monitors[i]
                if (m['left'] <= center_x < m['left'] + m['width'] and 
                    m['top'] <= center_y < m['top'] + m['height']):
                    target_mon = m
                    break
            
            rel_x, rel_y = ax - target_mon['left'], ay - target_mon['top']
            logging.debug(f"Overlay: монитор {target_mon['width']}x{target_mon['height']}+{target_mon['left']}+{target_mon['top']}")
        
        self.win.geometry(f"{aw}x{ah}+{rel_x}+{rel_y}")
        self.win.update_idletasks()
        self.win.deiconify()
        self.win.focus_set()
        
    def update(self, img: np.ndarray):
        pil = Image.fromarray(img)
        self.tk_img = ImageTk.PhotoImage(image=pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(width=img.shape[1], height=img.shape[0])
        
    def hide(self, event=None): 
        self.win.withdraw()


class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manga Translate Manager")
        self.root.geometry("900x650")
        self.root.configure(bg=Config.UI_BG_MAIN)
        self._style()
        
        logging.info("Инициализация приложения...")
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors): 
                logging.debug(f"Monitor {i}: {m}")
        
        self.ocr = OCRService(Config.OCR_LANGS)
        self.detector = BubbleDetector()
        self.translator = TranslationService(Config.API_KEY, Config.API_BASE_URL, Config.API_MODEL) if Config.API_KEY else None
        self.bubble_analyzer = BubbleAnalyzer(mode=Config.BUBBLE_MODE)
        self.selected_area = None
        self.processing = False
        self.overlay = ScreenOverlay(self.root)
        self.border_overlay = AreaBorderOverlay(self.root)
        self.last_result_crop = None
        self.current_hotkey_translate = Config.HOTKEY_TRANSLATE
        
        self._build_ui()
        self._bind_keys()
        
    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("S.TButton", background=Config.UI_BG_SIDEBAR, foreground=Config.UI_TEXT_MAIN, borderwidth=0, padding=12, font=("Segoe UI", 10))
        s.map("S.TButton", background=[("active", Config.UI_BG_BUTTON_ACTIVE)], foreground=[("active", Config.UI_ACCENT)])
        s.configure("A.TButton", background=Config.UI_BG_BUTTON, foreground=Config.UI_TEXT_MAIN, borderwidth=1, relief="flat", padding=10, font=("Segoe UI", 10, "bold"))
        s.map("A.TButton", background=[("active", Config.UI_ACCENT), ("pressed", "#00cc7a")], foreground=[("active","#000"), ("pressed","#000")])
        s.configure("TEntry", fieldbackground=Config.UI_BG_BUTTON, foreground=Config.UI_TEXT_MAIN, borderwidth=0, padding=5)
        
    def _build_ui(self):
        container = tk.Frame(self.root, bg=Config.UI_BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True)
        
        sb = tk.Frame(container, width=180, bg=Config.UI_BG_SIDEBAR, relief=tk.FLAT)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        
        tk.Label(sb, text="MANGA\nTRANSLATOR", bg=Config.UI_BG_SIDEBAR, fg=Config.UI_ACCENT, font=("Segoe UI", 14, "bold"), justify=tk.LEFT, pady=20).pack(fill=tk.X)
        
        nav = tk.Frame(sb, bg=Config.UI_BG_SIDEBAR)
        nav.pack(fill=tk.X, pady=10)
        self.btn_select = ttk.Button(nav, text="  Выделить область (F2)", style="S.TButton", command=self.start_select)
        self.btn_select.pack(fill=tk.X, pady=2)
        self.btn_translate = ttk.Button(nav, text=f"  Перевести ({Config.HOTKEY_TRANSLATE.upper()})", style="S.TButton", command=self.run_translate)
        self.btn_translate.pack(fill=tk.X, pady=2)
        
        content = tk.Frame(container, bg=Config.UI_BG_MAIN)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(content, text="Управление переводом", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_MAIN, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0,20))
        
        self.status = tk.StringVar(value="Готов. Выделите область на целевом мониторе.")
        tk.Label(content, textvariable=self.status, bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM, font=("Consolas", 9), wraplength=600).pack(anchor="w", fill=tk.X)
        
        tk.Label(content, text="Настройки API и Обработки", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(30,5))
        
        tk.Label(content, text="Провайдер:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w")
        provider_names = list(Config.PROVIDER_PRESETS.keys())
        self.e_provider_preset = ttk.Combobox(content, values=provider_names, state="readonly", width=37)
        self.e_provider_preset.set("Groq")
        self.e_provider_preset.bind("<<ComboboxSelected>>", self.select_provider_preset)
        self.e_provider_preset.pack(pady=5, fill=tk.X)
        
        tk.Label(content, text="API Key:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w", pady=(15,0))
        api_frame = tk.Frame(content, bg=Config.UI_BG_MAIN)
        api_frame.pack(pady=5, fill=tk.X)
        self.e_api = tk.Entry(api_frame, bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, insertbackground=Config.UI_ACCENT, relief=tk.FLAT, width=35, show="*")
        self.e_api.insert(0, Config.API_KEY)
        self.e_api.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_toggle_key = tk.Button(api_frame, text="Show", bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, command=self.toggle_api_key_visibility, relief=tk.FLAT, width=4)
        self.btn_toggle_key.pack(side=tk.RIGHT, padx=(5,0))
        
        tk.Label(content, text="API Base URL:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w", pady=(15,0))
        self.e_base_url = tk.Entry(content, bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, insertbackground=Config.UI_ACCENT, relief=tk.FLAT, width=40)
        self.e_base_url.insert(0, Config.API_BASE_URL)
        self.e_base_url.pack(pady=5, fill=tk.X)
        
        tk.Label(content, text="Model:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w", pady=(15,0))
        self.e_model = tk.Entry(content, bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, insertbackground=Config.UI_ACCENT, relief=tk.FLAT, width=40)
        self.e_model.insert(0, Config.API_MODEL)
        self.e_model.pack(pady=5, fill=tk.X)
        
        tk.Label(content, text="Путь к шрифту:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w", pady=(15,0))
        self.e_font = tk.Entry(content, bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, insertbackground=Config.UI_ACCENT, relief=tk.FLAT, width=40)
        self.e_font.insert(0, Config.FONT_PATH)
        self.e_font.pack(pady=5, fill=tk.X)
        
        tk.Label(content, text="Горячие клавиши", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(30,5))
        hotkey_frame = tk.Frame(content, bg=Config.UI_BG_MAIN)
        hotkey_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(hotkey_frame, text="Перевод:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM, width=15, anchor="w").pack(side=tk.LEFT)
        self.e_hotkey_translate = tk.Entry(hotkey_frame, bg=Config.UI_BG_BUTTON, fg=Config.UI_TEXT_MAIN, insertbackground=Config.UI_ACCENT, relief=tk.FLAT, width=10)
        self.e_hotkey_translate.insert(0, Config.HOTKEY_TRANSLATE)
        self.e_hotkey_translate.pack(side=tk.LEFT, padx=(5,0))
        
        ttk.Button(content, text="Применить настройки", style="A.TButton", command=self.apply_settings).pack(pady=25, anchor="w")
        
        tk.Label(content, text="Режим работы:", bg=Config.UI_BG_MAIN, fg=Config.UI_TEXT_DIM).pack(anchor="w", pady=(15,0))
        self.bubble_mode = tk.StringVar(value="manga")
        mode_frame = tk.Frame(content, bg=Config.UI_BG_MAIN)
        mode_frame.pack(anchor="w", pady=5)

        ttk.Radiobutton(mode_frame, text="Манга (круглые пузыри)", variable=self.bubble_mode, value="manga", command=self.on_mode_change).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Манхва (смешанные)", variable=self.bubble_mode, value="manhwa", command=self.on_mode_change).pack(anchor="w")

    def _bind_keys(self):
        def on_press(key):
            try:
                char = getattr(key, 'char', '')
                if char and self._should_trigger_hotkey(char, self.current_hotkey_translate):
                    self.root.after(0, self.run_translate)
                if key == keyboard.Key.f2:
                    self.root.after(0, self.start_select)
            except AttributeError:
                pass

        self.global_listener = keyboard.Listener(on_press=on_press)
        self.global_listener.daemon = True
        self.global_listener.start()

        if self.current_hotkey_translate:
            self.root.bind(f'<Key-{self.current_hotkey_translate.lower()}>', lambda e: self.run_translate())
            self.root.bind(f'<Key-{self.current_hotkey_translate.upper()}>', lambda e: self.run_translate())
        
        self.root.bind('<F2>', lambda e: self.start_select())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _should_trigger_hotkey(self, char, hotkey):
        if not char or not hotkey:
            return False
        if char.lower() == hotkey.lower():
            return True
        russian_char = self._get_russian_char(hotkey)
        return char.lower() == russian_char.lower()
    
    def _get_russian_char(self, char):
        mapping = {
            'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з',
            'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д',
            'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь'
        }
        return mapping.get(char.lower(), char)

    def start_select(self):
        if self.processing: 
            return
        self.root.iconify()
        time.sleep(0.15)
        sel = AreaSelector(self.root)
        area = sel.get()
        self.root.deiconify()
        if area:
            self.selected_area = area
            self.border_overlay.show(area)
            self.status.set(f"Область: {area[0]}x{area[1]} ({area[2]}x{area[3]})")
        else: 
            self.border_overlay.hide()
            self.status.set("Выделение отменено.")
            
    def run_translate(self):
        if self.processing:
            logging.debug("Обработка уже идет, игнорируем повторный вызов")
            return

        if not self.selected_area:
            self.status.set("Сначала выделите область (F2)!")
            return

        if not self.translator:
            messagebox.showwarning("Ошибка", "Укажите API Key в настройках")
            return

        if self.overlay.win.winfo_viewable():
            self.overlay.hide()
            self.last_result_crop = None
            self.status.set("Перевод скрыт. Повторное нажатие запустит новый захват.")
            return

        self.processing = True
        self.status.set("Захват и обработка...")
        threading.Thread(target=self._process, daemon=True).start()
        
    def _process(self):
        logging.info("Начало обработки изображения...")
        try:
            ax, ay, aw, ah = self.selected_area
            
            with mss.mss() as sct:
                virtual = sct.monitors[0]
                v_right = virtual['left'] + virtual['width']
                v_bottom = virtual['top'] + virtual['height']
                
                if ax < virtual['left'] or ay < virtual['top'] or ax+aw > v_right or ay+ah > v_bottom:
                    logging.warning("Область выходит за границы виртуального экрана, применяем обрезку.")
                    ax = max(virtual['left'], ax)
                    ay = max(virtual['top'], ay)
                    aw = min(v_right - ax, aw)
                    ah = min(v_bottom - ay, ah)
            
            with mss.mss() as sct:
                monitors = sct.monitors
                target_mon = None
                
                for i in range(1, len(monitors)):
                    m = monitors[i]
                    if (m['left'] <= ax < m['left'] + m['width'] and 
                        m['top'] <= ay < m['top'] + m['height']):
                        target_mon = m
                        break
                
                if target_mon is None:
                    target_mon = monitors[0]
                
                rel_x, rel_y = ax - target_mon['left'], ay - target_mon['top']
                full_img = np.array(sct.grab(target_mon))
                h_mon, w_mon = full_img.shape[:2]
                
                if full_img.shape[2] == 4:
                    full_img = cv2.cvtColor(full_img, cv2.COLOR_RGBA2RGB)
                elif full_img.shape[2] == 3:
                    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                    
                start_x, start_y = max(0, rel_x), max(0, rel_y)
                end_x, end_y = min(w_mon, rel_x + aw), min(h_mon, rel_y + ah)
                crop = full_img[start_y:end_y, start_x:end_x]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                raise ValueError(f"Область захвата пуста. Координаты: {ax},{ay} Размер: {aw}x{ah}")
                
            if crop.shape[2] == 4:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            if np.mean(crop) < 15:
                logging.error("Захваченное изображение черное! Отключите аппаратное ускорение в браузере.")
                self.root.after(0, lambda: self.status.set("Ошибка захвата (черный экран)."))
                self.processing = False
                return
                
            Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(Config.CACHE_DIR, "debug_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            bubbles = self.detector.detect(crop)
            
            expanded_bubbles = []
            expand = 8
            for bx1, by1, bx2, by2 in bubbles:
                exp_bx1 = max(0, bx1 - expand)
                exp_by1 = max(0, by1 - expand)
                exp_bx2 = min(crop.shape[1], bx2 + expand)
                exp_by2 = min(crop.shape[0], by2 + expand)
                expanded_bubbles.append((exp_bx1, exp_by1, exp_bx2, exp_by2))
            
            if Config.BUBBLE_MODE == 'manhwa':
                bubbles = normalize_overlapping_bubbles(expanded_bubbles, crop.shape)
            else:
                bubbles = expanded_bubbles
            
            valid_bubbles = []
            all_res = []

            for bx1, by1, bx2, by2 in bubbles:
                margin_x = int((bx2 - bx1) * 0.06)
                margin_y = int((by2 - by1) * 0.06)
                bx1_c, by1_c = max(0, bx1 + margin_x), max(0, by1 + margin_y)
                bx2_c, by2_c = min(crop.shape[1], bx2 - margin_x), min(crop.shape[0], by2 - margin_y)
                
                if bx2_c - bx1_c < 20 or by2_c - by1_c < 20:
                    continue

                valid_bubbles.append((bx1, by1, bx2, by2))
                sub_img = crop[by1_c:by2_c, bx1_c:bx2_c]
                sub_ocr = self.ocr.detect(sub_img)
                
                if not sub_ocr:
                    full_bubble = crop[max(0, by1):min(crop.shape[0], by2), max(0, bx1):min(crop.shape[1], bx2)]
                    if full_bubble.size > 0:
                        sub_ocr_full = self.ocr.detect(full_bubble)
                        if sub_ocr_full:
                            sub_ocr = []
                            for bbox, txt, conf in sub_ocr_full:
                                adjusted_bbox = [[p[0] + bx1, p[1] + by1] for p in bbox]
                                sub_ocr.append((adjusted_bbox, txt, conf))
                
                if not sub_ocr:
                    continue 
                
                for bbox, txt, conf in sub_ocr:
                    if isinstance(bbox, list) and bbox and bbox[0][0] >= bx1 and bbox[0][1] >= by1:
                        all_res.append((bbox, txt, conf))
                    else:
                        adjusted_bbox = [[p[0]+bx1_c, p[1]+by1_c] for p in bbox]
                        all_res.append((adjusted_bbox, txt, conf))
            
            res = filter_noise(deduplicate(all_res))
            if not res:
                self.root.after(0, lambda: self.status.set("Текст не обнаружен."))
                self.processing = False
                return
                
            final_bubbles = []
            b_texts = []
            
            if valid_bubbles:
                sorted_bubbles = sorted(valid_bubbles, key=lambda b: (b[1] // 50 * 50, b[0]))
                assignments = assign_ocr_to_bubbles(res, sorted_bubbles)
                
                for bubble_idx, (bx1, by1, bx2, by2) in enumerate(sorted_bubbles):
                    bubble_texts = assignments.get(bubble_idx, [])
                    final_bubbles.append((bx1, by1, bx2, by2))
                    
                    if bubble_texts:
                        bubble_texts.sort(key=lambda x: get_bbox_y(x[0]))
                        lines, cur_line = [], [bubble_texts[0]]
                        cur_y = get_bbox_y(bubble_texts[0][0])
                        
                        for bbox, txt, conf in bubble_texts[1:]:
                            y = get_bbox_y(bbox)
                            if abs(y - cur_y) > 12:
                                lines.append(' '.join(t for _, t, _ in cur_line))
                                cur_line = [(bbox, txt, conf)]
                                cur_y = y
                            else:
                                cur_line.append((bbox, txt, conf))
                        if cur_line: 
                            lines.append(' '.join(t for _, t, _ in cur_line))
                        b_texts.append('\n'.join(lines))
                    else:
                        b_texts.append("")

            outside_bubbles, outside_texts = [], []
            if Config.BUBBLE_MODE == 'manhwa':
                full_ocr = filter_noise(deduplicate(self.ocr.detect(crop)))
                outside_blocks = build_outside_text_boxes(full_ocr, final_bubbles, crop.shape)
                for blk in outside_blocks:
                    outside_bubbles.append(blk['rect'])
                    outside_texts.append(blk['text'])

            bubble_inputs = [clean_text(split_merged(t)) for t in b_texts]
            bubble_translated = self.translator.translate(bubble_inputs)
            bubble_translated = [postprocess(t) for t in bubble_translated]

            outside_inputs = [clean_text(split_merged(t)) for t in outside_texts]
            outside_translated = self.translator.translate(outside_inputs) if outside_inputs else []
            outside_translated = [postprocess(t) for t in outside_translated]

            render_bubbles = list(final_bubbles) + list(outside_bubbles)
            render_texts = list(bubble_translated) + list(outside_translated)
            render_original_texts = list(b_texts) + list(outside_texts)
            bubble_types = (['bubble'] * len(final_bubbles)) + (['outside'] * len(outside_bubbles))
            
            proc = crop.copy()
            text_boxes, collision_scores = compute_overlap_safe_text_boxes(render_bubbles, base_margin=10) if render_bubbles else ([], [])
            
            if Config.USE_WHITE_BOX and render_bubbles:
                fill_bubbles, fill_shapes = [], []
                for bubble, original_text, b_type in zip(render_bubbles, render_original_texts, bubble_types):
                    if not original_text.strip():
                        continue
                    fill_bubbles.append(bubble)
                    if b_type == 'outside':
                        fill_shapes.append('rectangular')
                    elif Config.BUBBLE_MODE == 'manga':
                        fill_shapes.append('round')
                    else:
                        fill_shapes.append(None)

                if fill_bubbles:
                    proc = fill_bubbles_adaptive(
                        proc, fill_bubbles, mode=Config.BUBBLE_MODE, forced_shapes=fill_shapes, analysis_img=crop
                    )

            for i, (bubble, text, b_type, original_text) in enumerate(zip(render_bubbles, render_texts, bubble_types, render_original_texts)):
                if not original_text.strip():
                    continue
                if text.strip():
                    bx1, by1, bx2, by2 = bubble
                    text_bbox = text_boxes[i] if i < len(text_boxes) else (bx1 + 10, by1 + 10, bx2 - 10, by2 - 10)
                    
                    if b_type == 'outside':
                        shape = 'rectangular'
                    elif Config.BUBBLE_MODE == 'manga':
                        shape = 'round'
                    else:
                        bubble_roi = crop[by1:by2, bx1:bx2] if 0 <= by1 < by2 <= crop.shape[0] and 0 <= bx1 < bx2 <= crop.shape[1] else None
                        if bubble_roi is not None and bubble_roi.size > 0:
                            shape, _ = self.bubble_analyzer.analyze_shape(bubble_roi, (bx1, by1, bx2, by2))
                        else:
                            shape = 'rectangular'
                    
                    collision_score = collision_scores[i] if i < len(collision_scores) else 0
                    dynamic_min_font = 10 if collision_score > 0 else Config.MIN_FONT_SIZE
                    proc = render_text_adaptive(
                        proc, text, text_bbox, bubble_shape=shape, min_font_size=dynamic_min_font
                    )
                
            self.last_result_crop = proc
            self.root.after(0, self._show_result, proc)
            logging.info("Обработка успешно завершена")
            
        except Exception as e:
            logging.error(f"Критическая ошибка обработки: {e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self.status.set(f"Ошибка: {msg}"))
            self.processing = False
            
    def _show_result(self, img):
        self.overlay.update(img)
        self.overlay.show(self.selected_area)
        self.status.set("Перевод применен. Кликните или нажмите Q для скрытия.")
        self.processing = False
            
    def apply_settings(self):
        Config.API_KEY = self.e_api.get().strip()
        Config.API_BASE_URL = self.e_base_url.get().strip()
        Config.API_MODEL = self.e_model.get().strip()
        Config.FONT_PATH = self.e_font.get().strip()
        
        Config.GROQ_API_KEY = Config.API_KEY
        Config.GROQ_MODEL = Config.API_MODEL
        
        new_hotkey_translate = self.e_hotkey_translate.get().strip().lower()
        
        if new_hotkey_translate != self.current_hotkey_translate:
            self.current_hotkey_translate = new_hotkey_translate
            Config.HOTKEY_TRANSLATE = new_hotkey_translate
            self.btn_translate.config(text=f"  Перевести ({new_hotkey_translate.upper()})")
            
            if hasattr(self, 'global_listener'):
                self.global_listener.stop()
            self._bind_keys()
            self.status.set("Настройки применены. Горячая клавиша обновлена.")
        else:
            self.translator = TranslationService(Config.API_KEY, Config.API_BASE_URL, Config.API_MODEL) if Config.API_KEY else None
            self.status.set("Настройки применены.")
    
    def select_provider_preset(self, event):
        preset_name = self.e_provider_preset.get()
        if preset_name in Config.PROVIDER_PRESETS:
            preset = Config.PROVIDER_PRESETS[preset_name]
            if preset["base_url"]:
                self.e_base_url.delete(0, tk.END)
                self.e_base_url.insert(0, preset["base_url"])
            self.status.set(f"Выбран провайдер: {preset_name}. Укажите модель вручную.")
    
    def toggle_api_key_visibility(self):
        if self.e_api.cget("show") == "*":
            self.e_api.config(show="")
            self.btn_toggle_key.config(text="Hide")
        else:
            self.e_api.config(show="*")
            self.btn_toggle_key.config(text="Show")
    
    def on_mode_change(self):
        Config.BUBBLE_MODE = self.bubble_mode.get()
        self.bubble_analyzer = BubbleAnalyzer(mode=Config.BUBBLE_MODE)
        self.status.set(f"Режим изменен: {Config.BUBBLE_MODE}")    
        
    def on_close(self):
        if hasattr(self, 'global_listener'):
            self.global_listener.stop()
        if self.overlay.win.winfo_exists(): 
            self.overlay.win.destroy()
        if self.border_overlay.win and self.border_overlay.win.winfo_exists(): 
            self.border_overlay.win.destroy()
        self.root.quit()
        
    def run(self):
        logging.info("Запуск Manga Translate Manager...")
        self.root.mainloop()

if __name__ == "__main__":
    MainApp().run()