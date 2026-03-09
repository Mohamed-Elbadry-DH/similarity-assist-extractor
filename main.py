"""
Similarity Assist — Python Extraction Service
=============================================
common.py is merged inline here so setuptools sees only one top-level module.
"""

from __future__ import annotations

import difflib
import io
import os
import re
import tempfile
import base64
import unicodedata
from typing import Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import fastapi
import fastapi.middleware.cors
import fastapi.responses
import numpy as np
import pytesseract
import requests
from fastapi import File, Form, UploadFile
from PIL import Image, ImageOps
from pytesseract import Output

# ---------------------------------------------------------------------------
# common.py — inlined verbatim
# ---------------------------------------------------------------------------

SCRIPT_RE = {
    'Arabic':   re.compile(r'[\u0600-\u06FF]'),
    'Cyrillic': re.compile(r'[\u0400-\u04FF]'),
    'Japanese': re.compile(r'[\u3040-\u30FF]'),
    'Han':      re.compile(r'[\u4E00-\u9FFF]'),
    'Korean':   re.compile(r'[\uAC00-\uD7AF]'),
    'Thai':     re.compile(r'[\u0E00-\u0E7F]'),
    'Latin':    re.compile(r'[A-Za-z]'),
}

OFFICE_LANG_MAP = {
    'FR':'fra','IT':'ita','ES':'spa','PT':'por','BR':'por','DE':'deu','VN':'vie','ID':'ind',
    'RU':'rus','EM':'eng','WO':'eng','US':'eng','GB':'eng','CN':'chi_sim','JP':'jpn','KR':'kor',
    'TH':'tha','AE':'ara','EG':'ara',
}

COLOR_NAMES = {
    'black':(0,0,0),'white':(255,255,255),'gray':(128,128,128),'silver':(192,192,192),
    'red':(220,20,60),'orange':(255,140,0),'yellow':(255,215,0),'green':(34,139,34),
    'olive':(128,128,0),'teal':(0,128,128),'cyan':(0,255,255),'blue':(30,144,255),
    'navy':(0,0,128),'purple':(128,0,128),'pink':(255,105,180),'brown':(139,69,19),
    'beige':(245,245,220),'gold':(212,175,55),
}

SPLIT_RE = re.compile(r'[\s,/|;:()\[\]{}\"""\'\'\'`~!@#$%^&*+=<>?-]+')


def canonicalize_brand_name(value: str) -> str:
    if value is None:
        return ''
    value = str(value).strip()
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace('&', ' AND ')
    value = re.sub(r'[\u2010-\u2015]', '-', value)
    value = re.sub(r"[^\w\s()'\/-]", ' ', value, flags=re.UNICODE)
    value = re.sub(r'\s+', ' ', value).strip().upper()
    return value


def clean_token(token: str) -> str:
    token = '' if token is None else str(token).strip()
    token = re.sub(
        r'^[^\w\u0600-\u06FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\u0E00-\u0E7F]+'
        r'|[^\w\u0600-\u06FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\u0E00-\u0E7F]+$',
        '', token,
    )
    return token


def tokenize_source(text: str) -> List[str]:
    return [t for t in (clean_token(p) for p in SPLIT_RE.split('' if text is None else str(text))) if t]


def normalize_latin_token(token: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', canonicalize_brand_name(token))


def guess_script(text: str) -> str:
    joined = '' if text is None else str(text)
    for name in ['Arabic', 'Korean', 'Japanese', 'Han', 'Cyrillic', 'Thai', 'Latin']:
        if SCRIPT_RE[name].search(joined):
            return name
    return ''


def build_ocr_languages(brand_name: str, ip_office: str = '') -> str:
    langs: List[str] = []
    script = guess_script(brand_name)
    script_lang = {
        'Arabic': 'ara', 'Cyrillic': 'rus', 'Japanese': 'jpn',
        'Han': 'chi_sim', 'Korean': 'kor', 'Thai': 'tha', 'Latin': 'eng',
    }.get(script)
    if script_lang:
        langs.append(script_lang)
    office_lang = OFFICE_LANG_MAP.get((ip_office or '').upper())
    if office_lang:
        langs.append(office_lang)
    langs.append('eng')
    out: List[str] = []
    for lang in langs:
        if lang not in out:
            out.append(lang)
    return '+'.join(out[:4])


def to_pg_text_array(values: Iterable[str]) -> str:
    escaped = []
    for value in values:
        s = str(value).replace('\\', '\\\\').replace('"', '\\"')
        escaped.append(f'"{s}"')
    return '{' + ','.join(escaped) + '}'


def to_pg_int_array(values: Iterable[int]) -> str:
    return '{' + ','.join(str(int(v)) for v in values) + '}'


def parse_nice_classes(value) -> List[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    out: List[int] = []
    for n in re.findall(r'\d+', text):
        i = int(n)
        if i not in out:
            out.append(i)
    return out


def _bits_to_hex(bits: np.ndarray) -> str:
    bitstr = ''.join('1' if int(b) else '0' for b in bits.flatten())
    return f'{int(bitstr, 2):0{len(bitstr)//4}x}'


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = np.array(image.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = gray > gray.mean()
    return _bits_to_hex(bits)


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = np.array(image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = gray[:, 1:] > gray[:, :-1]
    return _bits_to_hex(bits)


def perceptual_hash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    gray = np.array(image.convert('L').resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    dctlow = dct[:hash_size, :hash_size]
    med = float(np.median(dctlow[1:, 1:]))
    bits = dctlow > med
    return _bits_to_hex(bits)


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def dominant_colors(image: Image.Image, n_colors: int = 3) -> List[Tuple[str, float]]:
    rgb = image.convert('RGB')
    small = rgb.resize((96, 96))
    paletted = small.quantize(colors=max(n_colors, 3), method=Image.MEDIANCUT)
    palette = paletted.getpalette()
    counts = paletted.getcolors() or []
    total = sum(count for count, _ in counts) or 1
    out: List[Tuple[str, float]] = []
    for count, idx in sorted(counts, reverse=True)[:n_colors]:
        base = idx * 3
        hexv = '#%02X%02X%02X' % tuple(palette[base:base + 3])
        pct = round(count * 100.0 / total, 2)
        out.append((hexv, pct))
    while len(out) < n_colors:
        out.append((None, None))
    return out


def foreground_ratio(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float((thresh > 0).mean())
    dark_ratio = 1.0 - white_ratio
    return round(max(white_ratio, dark_ratio), 6)


def symmetry_score(gray: np.ndarray, axis: str = 'vertical') -> float:
    arr = gray.astype(np.float32) / 255.0
    if axis == 'vertical':
        half = arr[:, :arr.shape[1] // 2]
        mirror = np.fliplr(arr[:, -half.shape[1]:])
    else:
        half = arr[:arr.shape[0] // 2, :]
        mirror = np.flipud(arr[-half.shape[0]:, :])
    if half.size == 0 or mirror.size == 0:
        return 0.0
    score = 1.0 - float(np.abs(half - mirror).mean())
    return round(max(0.0, min(1.0, score)), 6)


def compute_v1_features(image_path: str) -> dict:
    image = Image.open(image_path)
    rgba = image.convert('RGBA')
    rgb = image.convert('RGB')
    gray_img = image.convert('L')
    gray = np.array(gray_img)
    rgb_arr = np.array(rgb)
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 100, 200)
    alpha = np.array(rgba)[..., 3]
    dom = dominant_colors(rgb, 3)
    return {
        'image_width':          int(rgb.width),
        'image_height':         int(rgb.height),
        'aspect_ratio':         round(rgb.width / rgb.height, 3) if rgb.height else None,
        'file_size_bytes':      int(os.path.getsize(image_path)),
        'has_transparency':     bool((alpha < 255).any()),
        'phash':                perceptual_hash(rgb),
        'dhash':                difference_hash(rgb),
        'ahash':                average_hash(rgb),
        'dominant_color_1_hex': dom[0][0],
        'dominant_color_1_pct': dom[0][1],
        'dominant_color_2_hex': dom[1][0],
        'dominant_color_2_pct': dom[1][1],
        'dominant_color_3_hex': dom[2][0],
        'dominant_color_3_pct': dom[2][1],
        'mean_red':             int(round(rgb_arr[..., 0].mean())),
        'mean_green':           int(round(rgb_arr[..., 1].mean())),
        'mean_blue':            int(round(rgb_arr[..., 2].mean())),
        'brightness_mean':      round(float(gray.mean()), 2),
        'brightness_std':       round(float(gray.std()), 2),
        'saturation_mean':      round(float(hsv[..., 1].mean()), 2),
        'edge_density':         round(float((edges > 0).mean()), 6),
        'entropy_score':        round(image_entropy(gray), 4),
        'foreground_ratio':     foreground_ratio(gray),
        'vertical_symmetry':    symmetry_score(gray, 'vertical'),
        'horizontal_symmetry':  symmetry_score(gray, 'horizontal'),
    }


def closest_color_name(hexv: str) -> str:
    try:
        hexv = str(hexv).lstrip('#')
        rgb = tuple(int(hexv[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return 'mixed'
    best = 'mixed'
    bestd = 10 ** 18
    for n, c in COLOR_NAMES.items():
        d = sum((rgb[i] - c[i]) ** 2 for i in range(3))
        if d < bestd:
            best = n
            bestd = d
    return best


def build_layout_type(row: dict, keyword_count: int) -> str:
    aspect = float(row.get('aspect_ratio') or 1)
    edge   = float(row.get('edge_density') or 0)
    vert   = float(row.get('vertical_symmetry') or 0)
    horiz  = float(row.get('horizontal_symmetry') or 0)
    if keyword_count == 0 and edge < 0.11:
        return 'icon-only-minimal'
    if keyword_count >= 6 and edge > 0.18:
        return 'text-heavy-label'
    if vert > 0.88 and horiz > 0.84:
        return 'balanced-emblem'
    if aspect < 0.85:
        return 'portrait-combined'
    if aspect > 1.15:
        return 'landscape-combined'
    return 'combined-standard'


def build_visual_tags(row: dict, keyword_count: int) -> List[str]:
    tags: List[str] = []
    aspect     = float(row.get('aspect_ratio') or 1)
    edge       = float(row.get('edge_density') or 0)
    entropy    = float(row.get('entropy_score') or 0)
    saturation = float(row.get('saturation_mean') or 0)
    vert       = float(row.get('vertical_symmetry') or 0)
    horiz      = float(row.get('horizontal_symmetry') or 0)
    tags.append('portrait' if aspect < 0.85 else 'landscape' if aspect > 1.15 else 'squareish')
    tags.append('no-readable-text' if keyword_count == 0 else 'light-text' if keyword_count < 4 else 'text-heavy')
    tags.append('busy-label-style' if edge > 0.18 and entropy > 6.5 else 'minimal' if edge < 0.09 else 'combined')
    if vert > 0.88 and horiz > 0.84:
        tags.append('high-symmetry')
    tags.append('muted' if saturation < 40 else 'high-color' if saturation > 120 else 'balanced-color')
    return tags


def build_text_presence_level(keyword_count: int, conf: float) -> str:
    if keyword_count == 0:
        return 'none'
    if keyword_count <= 2 or conf < 45:
        return 'low'
    if keyword_count <= 6 or conf < 70:
        return 'medium'
    return 'high'


def build_visual_description(row: dict, keywords: List[str], script_guess: str, conf: float) -> str:
    colors: List[str] = []
    for idx in [1, 2, 3]:
        hx  = row.get(f'dominant_color_{idx}_hex')
        pct = row.get(f'dominant_color_{idx}_pct')
        if hx not in ('', None) and pct not in ('', None):
            try:
                float(pct)
                colors.append(closest_color_name(hx))
            except Exception:
                pass
    color_text  = ', '.join(colors[:3]) if colors else 'mixed colors'
    aspect      = float(row.get('aspect_ratio') or 1)
    orientation = 'portrait-oriented' if aspect < 0.85 else 'landscape-oriented' if aspect > 1.15 else 'square or near-square'
    layout      = build_layout_type(row, len(keywords))
    text_text   = (
        'No reliable readable text was detected'
        if not keywords
        else f"Readable {(script_guess or 'mixed script').lower()} text includes {', '.join(keywords[:6])}"
    )
    style_map = {
        'text-heavy-label':     'a dense packaging-style or label-style composition',
        'balanced-emblem':      'a balanced emblem-like composition',
        'icon-only-minimal':    'a clean minimal graphic composition',
        'portrait-combined':    'a portrait combined logo with text and graphic elements',
        'landscape-combined':   'a landscape combined logo with text and graphic elements',
        'combined-standard':    'a combined text-and-graphic logo',
    }
    conf_note = (
        'high OCR confidence' if conf >= 70
        else 'moderate OCR confidence' if conf >= 45
        else 'low OCR confidence'
    )
    return (
        f"{orientation} logo dominated by {color_text}. "
        f"It appears as {style_map[layout]}. "
        f"{text_text}. "
        f"Extraction quality is {conf_note}."
    )


def merged_keywords(ocr_words: List[str], brand_name: str) -> Tuple[List[str], str]:
    source_tokens = [t for t in tokenize_source(brand_name) if len(t) > 1]
    source_norm   = [normalize_latin_token(t) for t in source_tokens]
    good_ocr: List[str] = []
    overlap = 0
    for tok in ocr_words:
        sc   = guess_script(tok)
        keep = len(tok) > 1 or sc not in ('', 'Latin')
        if keep:
            good_ocr.append(tok)
        nt = normalize_latin_token(tok)
        if nt and any(nt == s or nt in s or s in nt for s in source_norm if s):
            overlap += 1
    if source_tokens:
        if overlap >= 1:
            merged: List[str] = []
            for tok in good_ocr:
                nt = normalize_latin_token(tok)
                if guess_script(tok) != 'Latin' or any(nt == s or nt in s or s in nt for s in source_norm if s):
                    if tok not in merged:
                        merged.append(tok)
            for tok in source_tokens:
                if tok not in merged:
                    merged.append(tok)
            return merged[:20], 'ocr_plus_brand_fallback'
        if len(good_ocr) >= 3 and sum(len(t) for t in good_ocr) / max(len(good_ocr), 1) >= 4:
            uniq: List[str] = []
            for tok in good_ocr:
                if tok not in uniq:
                    uniq.append(tok)
            return uniq[:20], 'ocr_only'
        return source_tokens[:20], 'brand_fallback_only'
    uniq = []
    for tok in good_ocr:
        if tok not in uniq:
            uniq.append(tok)
    return uniq[:20], 'ocr_only'


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = fastapi.FastAPI(title="Similarity Assist Extractor", version="2.0.0")

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def fuzzy_correct(token: str, source_tokens: List[str]) -> str:
    if not re.search(r'[A-Za-z]', token):
        return token
    nt = normalize_latin_token(token)
    if len(nt) < 2:
        return token
    best       = token
    best_ratio = 0.0
    for source in source_tokens:
        if not re.search(r'[A-Za-z]', source):
            continue
        ns = normalize_latin_token(source)
        if len(ns) < 2:
            continue
        ratio = difflib.SequenceMatcher(None, nt, ns).ratio()
        if nt in ns or ns in nt:
            ratio = max(ratio, 0.9)
        if ratio > best_ratio:
            best_ratio = ratio
            best       = source
    return best if best_ratio >= 0.72 else token


def run_ocr(
    image: Image.Image,
    languages: str,
    source_tokens: List[str],
) -> Tuple[str, List[str], float | str]:
    gray = ImageOps.autocontrast(image.convert('L'))
    data = pytesseract.image_to_data(gray, lang=languages, config='--psm 6', output_type=Output.DICT)
    words:  List[str]   = []
    confs:  List[float] = []
    for text, conf in zip(data.get('text', []), data.get('conf', [])):
        token = clean_token(text)
        try:
            score = float(conf)
        except Exception:
            score = -1.0
        if not token or score < 25:
            continue
        token = fuzzy_correct(token, source_tokens)
        if re.fullmatch(r'[_\-=~—]+', token):
            continue
        if len(token) == 1 and not (token.isdigit() or guess_script(token)):
            continue
        if token not in words:
            words.append(token)
        confs.append(score)
    raw_text  = ' '.join(words)
    conf_mean: float | str = round(sum(confs) / len(confs), 2) if confs else ''
    return raw_text, words, conf_mean


# ---------------------------------------------------------------------------
# Full v2 extraction
# ---------------------------------------------------------------------------

def extract_full_features(
    image_bytes:         bytes,
    logo_filename:       str,
    record_id:           str,
    logo_url:            str  = '',
    brand_name_original: str  = '',
    ip_office:           str  = '',
    ocr_enabled:         bool = True,
) -> dict:
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(logo_filename)[1] or '.png',
        delete=False,
    ) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        base_row  = compute_v1_features(tmp_path)
        ocr_error = ''
        raw_text  = ''
        words:     List[str]       = []
        conf_mean: float | str     = ''
        languages  = build_ocr_languages(brand_name_original, ip_office)
        src_tokens = tokenize_source(brand_name_original)

        if ocr_enabled:
            try:
                img = Image.open(tmp_path)
                raw_text, words, conf_mean = run_ocr(img, languages, src_tokens)
            except Exception as exc:
                ocr_error = str(exc)[:250]

        kws, mode     = merged_keywords(words, brand_name_original)
        keyword_count = len(kws)
        script        = guess_script(' '.join(kws if kws else words))
        conf_float    = float(conf_mean) if conf_mean != '' else 0.0

        layout      = build_layout_type(base_row, keyword_count)
        tags        = build_visual_tags(base_row, keyword_count)
        description = build_visual_description(base_row, kws, script, conf_float)
        presence    = build_text_presence_level(keyword_count, conf_float)

        return {
            'record_id':           record_id,
            'logo_filename':       logo_filename,
            'logo_url':            logo_url,
            'ocr_text_raw':        raw_text,
            'ocr_keywords':        kws,
            'ocr_keyword_count':   keyword_count,
            'ocr_script_guess':    script,
            'ocr_confidence_mean': conf_mean,
            'ocr_languages_used':  languages,
            'text_source_mode':    mode,
            'text_presence_level': presence,
            'layout_type':         layout,
            'visual_tags':         tags,
            'visual_description':  description,
            'ocr_error':           ocr_error,
            'extraction_version':  'v2',
            **base_row,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Visual reranking helpers (SSIM + ORB)
# ---------------------------------------------------------------------------

def _decode_b64_image(b64: str) -> bytes:
    # supports raw base64 or data URLs
    if not b64:
        return b''
    b64 = str(b64).strip()
    if b64.startswith('data:') and ',' in b64:
        b64 = b64.split(',', 1)[1]
    return base64.b64decode(b64)


def _load_gray(img_bytes: bytes, size: int) -> np.ndarray:
    """Load bytes → grayscale uint8 resized to (size,size)."""
    im = Image.open(io.BytesIO(img_bytes))
    im = ImageOps.exif_transpose(im)
    im = im.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def _ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Standard SSIM (mean SSIM over full image) using Gaussian window."""
    # Ensure float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants (for 8-bit images)
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    score = float(np.mean(ssim_map))
    # clamp
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return round(score, 6)


def _orb_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """ORB keypoint matching score in [0,1]."""
    orb = cv2.ORB_create(nfeatures=600)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or not kp1 or not kp2:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    matches.sort(key=lambda m: m.distance)
    # "good" matches threshold (tuned)
    good = [m for m in matches if m.distance < 60]
    denom = max(20, min(len(kp1), len(kp2)))
    score = len(good) / float(denom)
    if score > 1:
        score = 1.0
    return round(float(score), 6)


def _fetch_url_bytes(url: str, timeout: int = 10) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={'User-Agent': 'SimilarityAssistExtractor/1.0'})
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Routes (Vercel strips /api/extract prefix automatically)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI mode — called by Next.js via child_process.spawn
# Usage: python3 backend/main.py --extract
#   stdin:  JSON line: {"image_b64": "...", "record_id": "...", "logo_filename": "...",
#                       "brand_name_original": "...", "ip_office": "...", "ocr_enabled": true}
#   stdout: JSON line: {"ok": true, "features": {...}}
#        or JSON line: {"ok": false, "error": "..."}
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import base64
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--extract':
        try:
            line = sys.stdin.readline()
            req  = json.loads(line)
            image_bytes = base64.b64decode(req['image_b64'])
            features    = extract_full_features(
                image_bytes         = image_bytes,
                logo_filename       = req.get('logo_filename', 'query.png'),
                record_id           = req.get('record_id', '_query_'),
                logo_url            = req.get('logo_url', ''),
                brand_name_original = req.get('brand_name_original', ''),
                ip_office           = req.get('ip_office', ''),
                ocr_enabled         = bool(req.get('ocr_enabled', True)),
            )
            print(json.dumps({'ok': True, 'features': features}), flush=True)
        except Exception as exc:
            print(json.dumps({'ok': False, 'error': str(exc)}), flush=True)
        sys.exit(0)

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)


@app.get('/health')
async def health() -> dict:
    ocr_available = False
    try:
        pytesseract.get_tesseract_version()
        ocr_available = True
    except Exception:
        pass
    return {'status': 'ok', 'ocr_available': ocr_available}


@app.post('/features')
async def extract_features(
    image:               UploadFile = File(...),
    record_id:           str        = Form(...),
    logo_filename:       str        = Form(''),
    logo_url:            str        = Form(''),
    brand_name_original: str        = Form(''),
    ip_office:           str        = Form(''),
    ocr_enabled:         str        = Form('true'),
):
    image_bytes = await image.read()
    filename    = logo_filename or image.filename or 'logo.png'
    do_ocr      = ocr_enabled.lower() != 'false'
    try:
        features = extract_full_features(
            image_bytes=image_bytes,
            logo_filename=filename,
            record_id=record_id,
            logo_url=logo_url,
            brand_name_original=brand_name_original,
            ip_office=ip_office,
            ocr_enabled=do_ocr,
        )
        return fastapi.responses.JSONResponse(content={'features': features, 'ok': True})
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=500,
        )


@app.post('/reprocess')
async def reprocess_from_url(body: dict):
    import urllib.request
    image_url:           str  = body.get('image_url', '')
    record_id:           str  = body.get('record_id', '')
    logo_filename:       str  = body.get('logo_filename', '')
    logo_url:            str  = body.get('logo_url', image_url)
    brand_name_original: str  = body.get('brand_name_original', '')
    ip_office:           str  = body.get('ip_office', '')
    do_ocr:              bool = body.get('ocr_enabled', True)
    if not image_url or not record_id:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': 'image_url and record_id are required'},
            status_code=400,
        )
    try:
        blob_token = os.environ.get('BLOB_READ_WRITE_TOKEN', '')
        req = urllib.request.Request(
            image_url,
            headers={'Authorization': f'Bearer {blob_token}'} if blob_token else {},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            image_bytes = resp.read()
        features = extract_full_features(
            image_bytes=image_bytes,
            logo_filename=logo_filename or os.path.basename(image_url),
            record_id=record_id,
            logo_url=logo_url,
            brand_name_original=brand_name_original,
            ip_office=ip_office,
            ocr_enabled=do_ocr,
        )
        return fastapi.responses.JSONResponse(content={'features': features, 'ok': True})
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=500,
        )


@app.post('/rerank')
async def rerank(body: dict):
    """Second-stage visual reranking (SSIM + ORB).

    Request JSON:
      {
        "query_image": "<base64 or dataURL>",
        "candidates": [{"record_id": "...", "logo_url": "..."}, ...],
        "top_k": 30,
        "use_orb": true
      }

    Response JSON:
      {"ok": true, "results": [{"record_id":..., "ssim_score":..., "orb_score":...}, ...]}
    """
    try:
        query_b64 = body.get('query_image', '')
        candidates = body.get('candidates', []) or []
        top_k = int(body.get('top_k', 30) or 30)
        use_orb = bool(body.get('use_orb', True))
        if not query_b64 or not isinstance(candidates, list) or not candidates:
            return fastapi.responses.JSONResponse(
                content={'ok': False, 'error': 'query_image and candidates are required'},
                status_code=400,
            )

        query_bytes = _decode_b64_image(query_b64)
        # SSIM uses small grayscale
        q_gray_64 = _load_gray(query_bytes, 64)
        # ORB uses larger grayscale
        q_gray_256 = _load_gray(query_bytes, 256) if use_orb else None

        results = []
        # limit
        candidates = candidates[:max(1, min(top_k, 60))]

        def work(c: dict):
            rid = str(c.get('record_id', '') or '')
            url = str(c.get('logo_url', '') or '')
            if not rid or not url:
                return {'record_id': rid, 'ssim_score': 0.0, 'orb_score': 0.0, 'error': 'missing record_id or logo_url'}
            try:
                img_bytes = _fetch_url_bytes(url, timeout=12)
                c_gray_64 = _load_gray(img_bytes, 64)
                ssim = _ssim_score(q_gray_64, c_gray_64)
                orb = 0.0
                if use_orb and q_gray_256 is not None:
                    c_gray_256 = _load_gray(img_bytes, 256)
                    orb = _orb_score(q_gray_256, c_gray_256)
                return {'record_id': rid, 'ssim_score': ssim, 'orb_score': orb}
            except Exception as exc:
                return {'record_id': rid, 'ssim_score': 0.0, 'orb_score': 0.0, 'error': str(exc)}

        # run in parallel (safe for I/O)
        max_workers = int(os.environ.get('RERANK_WORKERS', '6'))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for out in ex.map(work, candidates):
                results.append(out)

        return fastapi.responses.JSONResponse(content={'ok': True, 'results': results})
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=500,
        )
