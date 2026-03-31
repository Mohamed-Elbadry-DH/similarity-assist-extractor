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
import unicodedata
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, List, Tuple

import cv2
import fastapi
import fastapi.middleware.cors
import fastapi.responses
import numpy as np
import pytesseract
from fastapi import File, Form, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
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


def _normalized_token_length(token: str) -> int:
    token = normalize_latin_token(token) or clean_token(token)
    return len(token or '')


def is_likely_single_wordmark(keywords: List[str], conf: float) -> bool:
    cleaned = [k for k in (keywords or []) if clean_token(k)]
    if len(cleaned) != 1:
        return False
    longest = _normalized_token_length(cleaned[0])
    return longest >= 5 and conf >= 40


def build_text_presence_level(keywords: List[str], conf: float) -> str:
    keyword_count = len(keywords or [])
    if keyword_count == 0:
        return 'none'

    longest = max((_normalized_token_length(k) for k in (keywords or [])), default=0)
    total_len = sum(_normalized_token_length(k) for k in (keywords or []))

    # Stylized single-word wordmarks (e.g. POLARIS) often OCR as just one token.
    # They should not be downgraded to 'low' automatically when confidence is decent.
    if is_likely_single_wordmark(keywords, conf):
        if conf >= 72 or longest >= 8:
            return 'high'
        return 'medium'

    # Two-token wordmarks can still be clearly text-led when OCR is readable enough.
    if keyword_count == 2 and total_len >= 8 and conf >= 45:
        return 'medium' if conf < 72 else 'high'

    if keyword_count <= 2 or conf < 45:
        return 'low'
    if keyword_count <= 6 or conf < 70:
        return 'medium'
    return 'high'


def build_visual_description(
    row: dict,
    keywords: List[str],
    script_guess: str,
    conf: float,
    brand_hint_tokens: List[str] | None = None,
    text_source_mode: str = 'none',
) -> str:
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
    if not keywords:
        if brand_hint_tokens and text_source_mode == 'brand_fallback_only':
            text_text = 'No reliable readable text was detected. Request metadata provided a brand hint: ' + ', '.join(brand_hint_tokens[:4])
        else:
            text_text = 'No reliable readable text was detected'
    else:
        text_text = f"Readable {(script_guess or 'mixed script').lower()} text includes {', '.join(keywords[:6])}"
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


class InputValidationError(ValueError):
    pass


MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_BYTES', str(10 * 1024 * 1024)))
MAX_DOWNLOAD_BYTES = int(os.environ.get('MAX_DOWNLOAD_BYTES', str(MAX_UPLOAD_BYTES)))
MAX_IMAGE_PIXELS = int(os.environ.get('MAX_IMAGE_PIXELS', '25000000'))
ALLOWED_UPLOAD_CONTENT_TYPES = {
    'image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp', 'image/tiff', 'image/gif',
}
ALLOWED_IMAGE_FORMATS = {'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF', 'GIF'}
FETCH_USER_AGENT = os.environ.get('FETCH_USER_AGENT', 'SimilarityAssist/1.1')
ALLOWED_FETCH_HOSTS = [h.strip().lower() for h in os.environ.get('ALLOWED_FETCH_HOSTS', '').split(',') if h.strip()]
ALLOW_PRIVATE_NETWORK_FETCH = os.environ.get('ALLOW_PRIVATE_NETWORK_FETCH', '').lower() in {'1', 'true', 'yes'}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def merge_text_sources(ocr_words: List[str], brand_name: str) -> dict:
    source_tokens = [t for t in tokenize_source(brand_name) if len(t) > 1]
    source_norm = [normalize_latin_token(t) for t in source_tokens]

    unique_ocr: List[str] = []
    overlap_tokens: List[str] = []

    for tok in ocr_words:
        sc = guess_script(tok)
        keep = len(tok) > 1 or sc not in ('', 'Latin')
        if not keep:
            continue
        if tok not in unique_ocr:
            unique_ocr.append(tok)

        nt = normalize_latin_token(tok)
        if nt and any(nt == s or nt in s or s in nt for s in source_norm if s):
            if tok not in overlap_tokens:
                overlap_tokens.append(tok)

    if unique_ocr and source_tokens:
        mode = 'ocr_plus_brand_fallback' if overlap_tokens else 'ocr_only'
    elif unique_ocr:
        mode = 'ocr_only'
    elif source_tokens:
        mode = 'brand_fallback_only'
    else:
        mode = 'none'

    return {
        'ocr_keywords': unique_ocr[:20],
        'brand_name_input_tokens': source_tokens[:20],
        'brand_name_overlap_keywords': overlap_tokens[:20],
        'text_source_mode': mode,
    }


def _safe_filename(name: str) -> str:
    raw = os.path.basename(str(name or 'image'))
    return raw or 'image'


def _validate_upload_content_type(upload: UploadFile) -> None:
    content_type = (upload.content_type or '').lower().strip()
    if content_type and content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise InputValidationError(f'Unsupported content type: {content_type}')


def _validate_image_bytes(image_bytes: bytes, filename: str = 'image') -> tuple[int, int, str]:
    if not image_bytes:
        raise InputValidationError('Empty image payload')
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InputValidationError(f'Image is too large ({len(image_bytes)} bytes). Max allowed is {MAX_UPLOAD_BYTES} bytes')

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            image_format = (probe.format or '').upper()
            probe = ImageOps.exif_transpose(probe)
            probe.load()
            width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise InputValidationError(f'Image exceeds safe pixel limit: {exc}') from exc
    except UnidentifiedImageError as exc:
        raise InputValidationError(f'Unsupported or corrupted image: {filename}') from exc
    except Exception as exc:
        raise InputValidationError(f'Cannot decode image {filename}: {exc}') from exc

    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise InputValidationError(f'Unsupported image format: {image_format or "unknown"}')
    if width <= 0 or height <= 0:
        raise InputValidationError('Invalid image dimensions')
    if width * height > MAX_IMAGE_PIXELS:
        raise InputValidationError(f'Image dimensions exceed safe pixel limit ({MAX_IMAGE_PIXELS} pixels)')

    return width, height, image_format


def _is_public_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or
        ip.is_reserved or ip.is_unspecified
    )


def _is_allowed_host(hostname: str) -> None:
    host = (hostname or '').strip().lower().rstrip('.')
    if not host:
        raise InputValidationError('Missing remote hostname')
    if host in {'localhost', 'localhost.localdomain'}:
        raise InputValidationError('Localhost fetch is not allowed')

    if ALLOWED_FETCH_HOSTS:
        if not any(host == allowed or host.endswith('.' + allowed) for allowed in ALLOWED_FETCH_HOSTS):
            raise InputValidationError(f'Host not allowed for remote fetch: {host}')

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise InputValidationError(f'Cannot resolve remote host: {host}') from exc

    resolved_ips = {info[4][0] for info in infos if info and info[4]}
    if not resolved_ips:
        raise InputValidationError(f'Cannot resolve remote host: {host}')

    if not ALLOW_PRIVATE_NETWORK_FETCH:
        for ip_str in resolved_ips:
            if not _is_public_ip(ip_str):
                raise InputValidationError(f'Remote host resolves to a private or restricted address: {host}')


def _validate_remote_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or '').strip())
    if parsed.scheme not in {'http', 'https'}:
        raise InputValidationError('Remote URL must use http or https')
    if parsed.username or parsed.password:
        raise InputValidationError('Remote URL credentials are not allowed')
    _is_allowed_host(parsed.hostname or '')
    return parsed


def _read_limited_response(resp, limit_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = resp.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise InputValidationError(f'Remote file exceeds max allowed size ({limit_bytes} bytes)')
        chunks.append(chunk)
    return b''.join(chunks)


def _safe_read_url_bytes(url: str, timeout: int = 15, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    parsed = _validate_remote_url(url)
    req = urllib.request.Request(
        parsed.geturl(),
        headers={'User-Agent': FETCH_USER_AGENT},
        method='GET',
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get_content_type() or '').lower()
            if content_type and content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
                raise InputValidationError(f'Unsupported remote content type: {content_type}')
            content_length = resp.headers.get('Content-Length')
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise InputValidationError(f'Remote file exceeds max allowed size ({max_bytes} bytes)')
                except ValueError:
                    pass
            data = _read_limited_response(resp, max_bytes)
    except urllib.error.HTTPError as exc:
        raise exc
    except urllib.error.URLError as exc:
        raise InputValidationError(f'Failed to fetch remote image: {exc.reason}') from exc

    _validate_image_bytes(data, parsed.path or 'remote-image')
    return data


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
    _validate_image_bytes(image_bytes, logo_filename or 'logo')

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

        text_meta = merge_text_sources(words, brand_name_original)
        kws = text_meta['ocr_keywords']
        brand_hint_tokens = text_meta['brand_name_input_tokens']
        brand_overlap_keywords = text_meta['brand_name_overlap_keywords']
        mode = text_meta['text_source_mode']
        keyword_count = len(kws)
        script = guess_script(' '.join(kws if kws else words))
        conf_float = float(conf_mean) if conf_mean != '' else 0.0

        layout = build_layout_type(base_row, keyword_count)
        tags = build_visual_tags(base_row, keyword_count)
        description = build_visual_description(
            base_row,
            kws,
            script,
            conf_float,
            brand_hint_tokens=brand_hint_tokens,
            text_source_mode=mode,
        )
        presence = build_text_presence_level(kws, conf_float)

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
            'brand_name_input_tokens': brand_hint_tokens,
            'brand_name_overlap_keywords': brand_overlap_keywords,
            'extraction_version':  'v2.1',
            **base_row,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


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
    filename = logo_filename or image.filename or 'logo.png'
    do_ocr = ocr_enabled.lower() != 'false'
    try:
        _validate_upload_content_type(image)
        image_bytes = await image.read()
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
    except InputValidationError as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=400,
        )
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=500,
        )


@app.post('/reprocess')
async def reprocess_from_url(body: dict):
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
        image_bytes = _safe_read_url_bytes(image_url, timeout=30)
        features = extract_full_features(
            image_bytes=image_bytes,
            logo_filename=logo_filename or _safe_filename(image_url),
            record_id=record_id,
            logo_url=logo_url,
            brand_name_original=brand_name_original,
            ip_office=ip_office,
            ocr_enabled=do_ocr,
        )
        return fastapi.responses.JSONResponse(content={'features': features, 'ok': True})
    except InputValidationError as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=400,
        )
    except urllib.error.HTTPError as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': f'http_error_{exc.code}'},
            status_code=502,
        )
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={'ok': False, 'error': str(exc)},
            status_code=500,
        )


def _prepare_logo_variants(img_bytes: bytes, thumb_size=(192, 192)):
    """
    Returns:
      full_arr: grayscale logo centered in white square
      crop_arr: whitespace-trimmed grayscale logo centered in white square
      edge_arr: edge map for shape/structure comparison
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)

    gray = bg.convert("L")
    arr = np.array(gray)

    mask = arr < 245
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        pad_r = max(2, int((r1 - r0 + 1) * 0.06))
        pad_c = max(2, int((c1 - c0 + 1) * 0.06))

        r0 = max(0, r0 - pad_r)
        r1 = min(arr.shape[0] - 1, r1 + pad_r)
        c0 = max(0, c0 - pad_c)
        c1 = min(arr.shape[1] - 1, c1 + pad_c)

        cropped = gray.crop((c0, r0, c1 + 1, r1 + 1))
    else:
        cropped = gray

    full_contained = ImageOps.contain(gray, thumb_size, Image.Resampling.LANCZOS)
    full_canvas = Image.new("L", thumb_size, 255)
    fx = (thumb_size[0] - full_contained.size[0]) // 2
    fy = (thumb_size[1] - full_contained.size[1]) // 2
    full_canvas.paste(full_contained, (fx, fy))

    crop_contained = ImageOps.contain(cropped, thumb_size, Image.Resampling.LANCZOS)
    crop_canvas = Image.new("L", thumb_size, 255)
    cx = (thumb_size[0] - crop_contained.size[0]) // 2
    cy = (thumb_size[1] - crop_contained.size[1]) // 2
    crop_canvas.paste(crop_contained, (cx, cy))

    full_arr = np.array(full_canvas, dtype=np.float32)
    crop_arr = np.array(crop_canvas, dtype=np.float32)

    edge_arr = cv2.Canny(crop_arr.astype(np.uint8), 80, 160)

    return full_arr, crop_arr, edge_arr


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = 6.5025, 58.5225

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)

    ssim_map = num / (den + 1e-10)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _orb_similarity(query_u8: np.ndarray, cand_u8: np.ndarray):
    orb = cv2.ORB_create(1200)
    _, q_desc = orb.detectAndCompute(query_u8, None)
    _, c_desc = orb.detectAndCompute(cand_u8, None)

    if q_desc is None or c_desc is None or len(q_desc) == 0 or len(c_desc) == 0:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(q_desc, c_desc, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    max_possible = min(len(q_desc), len(c_desc))
    if max_possible <= 0:
        return None

    return float(np.clip(len(good) / max_possible, 0.0, 1.0))


def _edge_overlap_score(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    a = (edge_a > 0).astype(np.uint8)
    b = (edge_b > 0).astype(np.uint8)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(np.clip(inter / union, 0.0, 1.0))


def _visual_score(ssim_full, ssim_crop, orb_score, edge_score):
    values = []

    if ssim_full is not None:
        values.append(("ssim_full", ssim_full))
    if ssim_crop is not None:
        values.append(("ssim_crop", ssim_crop))
    if orb_score is not None:
        values.append(("orb", orb_score))
    if edge_score is not None:
        values.append(("edge", edge_score))

    if not values:
        return None

    weights = {
        "ssim_full": 0.18,
        "ssim_crop": 0.32,
        "orb": 0.32,
        "edge": 0.18,
    }

    total = 0.0
    denom = 0.0
    for name, v in values:
        total += weights[name] * float(v)
        denom += weights[name]

    if denom <= 0:
        return None

    return float(np.clip(total / denom, 0.0, 1.0))



@app.post("/rerank")
async def rerank_candidates(
    image: UploadFile = File(...),
    candidates: str = Form(...),
):
    """
    Visual reranker.

    Important:
    - returns usable=False for invalid/unusable comparisons
    - does NOT use fake 0/0 scores as evidence
    - compares both full normalized logo and trimmed-content logo
    """
    import json
    import urllib.error

    try:
        candidate_list = json.loads(candidates)
    except json.JSONDecodeError as exc:
        return fastapi.responses.JSONResponse(
            content={"ok": False, "error": f"Invalid candidates JSON: {exc}"},
            status_code=400,
        )

    try:
        _validate_upload_content_type(image)
        query_bytes = await image.read()
        _validate_image_bytes(query_bytes, image.filename or 'query.png')
        q_full, q_crop, q_edge = _prepare_logo_variants(query_bytes)
    except Exception as exc:
        return fastapi.responses.JSONResponse(
            content={"ok": False, "error": f"Cannot decode query image: {exc}"},
            status_code=400,
        )

    results = []

    for item in candidate_list:
        record_id = item.get("record_id", "")
        logo_url = item.get("logo_url", "")

        if not logo_url or not str(logo_url).startswith("http"):
            results.append({
                "record_id": record_id,
                "usable": False,
                "ssim_score": None,
                "orb_score": None,
                "edge_score": None,
                "visual_score": None,
                "error": "missing_or_invalid_logo_url",
            })
            continue

        try:
            cand_bytes = _safe_read_url_bytes(logo_url, timeout=15)
            c_full, c_crop, c_edge = _prepare_logo_variants(cand_bytes)

            ssim_full = _ssim_score(q_full, c_full)
            ssim_crop = _ssim_score(q_crop, c_crop)

            orb_score = _orb_similarity(
                q_crop.astype(np.uint8),
                c_crop.astype(np.uint8),
            )

            edge_score = _edge_overlap_score(q_edge, c_edge)

            visual_score = _visual_score(
                ssim_full=ssim_full,
                ssim_crop=ssim_crop,
                orb_score=orb_score,
                edge_score=edge_score,
            )

            usable = visual_score is not None

            results.append({
                "record_id": record_id,
                "usable": usable,
                "ssim_score": ssim_crop,
                "orb_score": orb_score,
                "edge_score": edge_score,
                "visual_score": visual_score,
            })

        except InputValidationError as exc:
            results.append({
                "record_id": record_id,
                "usable": False,
                "ssim_score": None,
                "orb_score": None,
                "edge_score": None,
                "visual_score": None,
                "error": str(exc),
            })
        except urllib.error.HTTPError as exc:
            results.append({
                "record_id": record_id,
                "usable": False,
                "ssim_score": None,
                "orb_score": None,
                "edge_score": None,
                "visual_score": None,
                "error": f"http_error_{exc.code}",
            })
        except Exception as exc:
            results.append({
                "record_id": record_id,
                "usable": False,
                "ssim_score": None,
                "orb_score": None,
                "edge_score": None,
                "visual_score": None,
                "error": str(exc),
            })

    return fastapi.responses.JSONResponse(
        content={
            "ok": True,
            "results": results,
        }
    )


