"""
Efficient CSS color parsing and conversion with RGBA as the hub.
Supported: named, transparent, currentColor, hex 3/4/6/8, rgb/rgba (modern/legacy),
hsl/hsla (modern/legacy), hwb, lab, lch, oklab, oklch.
Excludes: color() wide-gamut spaces (display-p3, rec2020, etc.).
"""

import re
import math
from typing import Dict, Optional, Tuple, Union, Literal
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, Field

from schemas.requests import (
    ColorConvertRequest,
)
from schemas.responses import SuccessResponse

# Type definitions
class RGBA(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)
    a: float = Field(ge=0.0, le=1.0)

TargetSpace = Literal["hex", "rgb", "hsl", "hwb", "lab", "lch", "oklab", "oklch", "named"]

# Minimal named list (complete CSS Color 4 set can be inserted if needed)
NAMED: Dict[str, str] = {
    "black": "#000000",
    "silver": "#c0c0c0",
    "gray": "#808080",
    "white": "#ffffff",
    "maroon": "#800000",
    "red": "#ff0000",
    "purple": "#800080",
    "fuchsia": "#ff00ff",
    "green": "#008000",
    "lime": "#00ff00",
    "olive": "#808000",
    "yellow": "#ffff00",
    "navy": "#000080",
    "blue": "#0000ff",
    "teal": "#008080",
    "aqua": "#00ffff",
    "orange": "#ffa500",
    "rebeccapurple": "#663399",
    "transparent": "#00000000",
}

# Regular expression patterns
ws = r"\s*"
num = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
perc = f"{num}%"
angle = f"{num}(?:deg|grad|rad|turn)?"
slash = f"{ws}\/{ws}"
comma = f"{ws},{ws}"
sep = f"{ws}"

def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))

def pct(x: str) -> float:
    """Convert percentage string to decimal."""
    return float(x.replace("%", "")) / 100

def angle_to_deg(s: str) -> float:
    """Convert angle string to degrees."""
    m = re.match(f"^({num})(deg|grad|rad|turn)?$", s, re.IGNORECASE)
    if not m:
        return 0
    v = float(m.group(1))
    unit = (m.group(2) or "deg").lower()
    if unit == "deg":
        return v
    elif unit == "grad":
        return (v * 9) / 10
    elif unit == "rad":
        return (v * 180) / math.pi
    elif unit == "turn":
        return v * 360
    else:
        return v

# HEX -------------------------------------------------------------

HEX_RE = re.compile(r"^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$", re.IGNORECASE)

def parse_hex(s: str) -> Optional[RGBA]:
    """Parse hex color string to RGBA."""
    m = HEX_RE.match(s)
    if not m:
        return None
    h = m.group(1)
    if len(h) in [3, 4]:
        r = int(h[0] + h[0], 16)
        g = int(h[1] + h[1], 16)
        b = int(h[2] + h[2], 16)
        a = 1
        if len(h) == 4:
            a = int(h[3] + h[3], 16) / 255
    else:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        a = 1
        if len(h) == 8:
            a = int(h[6:8], 16) / 255
    return RGBA(r=r, g=g, b=b, a=a)

# RGB -------------------------------------------------------------

RGB_MODERN_RE = re.compile(
    f"^rgb{ws}\\({ws}({num}%?|none)(?:{sep}|{comma})({num}%?|none)(?:{sep}|{comma})({num}%?|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
    re.IGNORECASE
)

RGBA_LEGACY_RE = re.compile(
    f"^rgba{ws}\\({ws}({num}%?){comma}({num}%?){comma}({num}%?){comma}({num}|{perc}){ws}\\)$",
    re.IGNORECASE
)

def parse_rgb(s: str) -> Optional[RGBA]:
    """Parse RGB/RGBA color string to RGBA."""
    m = RGB_MODERN_RE.match(s)
    if m:
        R, G, B, A = m.groups()
        def cv(t: str) -> int:
            if t == "none":
                return 0
            elif t.endswith("%"):
                return int(clamp(round(pct(t) * 255), 0, 255))
            else:
                return int(clamp(round(float(t)), 0, 255))
        
        r = cv(R)
        g = cv(G)
        b = cv(B)
        a = 1 if A is None or A == "none" else (
            clamp(pct(A), 0, 1) if A.endswith("%") else clamp(float(A), 0, 1)
        )
        return RGBA(r=r, g=g, b=b, a=a)
    
    m = RGBA_LEGACY_RE.match(s)
    if m:
        r_val, g_val, b_val, a_val = m.groups()
        r = int(clamp(round(pct(r_val) * 255), 0, 255)) if r_val.endswith("%") else int(clamp(round(float(r_val)), 0, 255))
        g = int(clamp(round(pct(g_val) * 255), 0, 255)) if g_val.endswith("%") else int(clamp(round(float(g_val)), 0, 255))
        b = int(clamp(round(pct(b_val) * 255), 0, 255)) if b_val.endswith("%") else int(clamp(round(float(b_val)), 0, 255))
        a = clamp(pct(a_val), 0, 1) if a_val.endswith("%") else clamp(float(a_val), 0, 1)
        return RGBA(r=r, g=g, b=b, a=a)
    
    return None

# HSL -------------------------------------------------------------

HSL_RE = re.compile(
    f"^hsl{ws}\\({ws}({angle}|none)(?:{sep}|{comma})({perc}|none)(?:{sep}|{comma})({perc}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
    re.IGNORECASE
)

HSLA_RE = re.compile(
    f"^hsla{ws}\\({ws}({angle}){comma}({perc}){comma}({perc}){comma}({num}|{perc}){ws}\\)$",
    re.IGNORECASE
)

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL to RGB. h in deg, s,l in [0,1]."""
    h = ((h % 360) + 360) % 360
    c = (1 - abs(2 * l - 1)) * s
    hp = h / 60
    x = c * (1 - abs((hp % 2) - 1))
    
    if 0 <= hp < 1:
        r1, g1, b1 = c, x, 0
    elif 1 <= hp < 2:
        r1, g1, b1 = x, c, 0
    elif 2 <= hp < 3:
        r1, g1, b1 = 0, c, x
    elif 3 <= hp < 4:
        r1, g1, b1 = 0, x, c
    elif 4 <= hp < 5:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    
    m = l - c / 2
    return (
        int(round((r1 + m) * 255)),
        int(round((g1 + m) * 255)),
        int(round((b1 + m) * 255))
    )

def parse_hsl(s: str) -> Optional[RGBA]:
    """Parse HSL/HSLA color string to RGBA."""
    m = HSL_RE.match(s)
    if m:
        h_val, s_val, l_val, a_val = m.groups()
        h = 0 if h_val == "none" else angle_to_deg(h_val)
        sv = 0 if s_val == "none" else clamp(pct(s_val), 0, 1)
        lv = 0 if l_val == "none" else clamp(pct(l_val), 0, 1)
        a = 1 if a_val is None or a_val == "none" else (
            clamp(pct(a_val), 0, 1) if a_val.endswith("%") else clamp(float(a_val), 0, 1)
        )
        r, g, b = hsl_to_rgb(h, sv, lv)
        return RGBA(r=r, g=g, b=b, a=a)
    
    m = HSLA_RE.match(s)
    if m:
        h_val, s_val, l_val, a_val = m.groups()
        h = angle_to_deg(h_val)
        sV = clamp(pct(s_val), 0, 1)
        lV = clamp(pct(l_val), 0, 1)
        a = clamp(pct(a_val), 0, 1) if a_val.endswith("%") else clamp(float(a_val), 0, 1)
        r, g, b = hsl_to_rgb(h, sV, lV)
        return RGBA(r=r, g=g, b=b, a=a)
    
    return None

# HWB -------------------------------------------------------------

HWB_RE = re.compile(
    f"^hwb{ws}\\({ws}({angle}|none)(?:{sep}|{comma})({perc}|none)(?:{sep}|{comma})({perc}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
    re.IGNORECASE
)

def hwb_to_rgb(h: float, w: float, bl: float) -> Tuple[int, int, int]:
    """Convert HWB to RGB."""
    # CSS spec: convert H to RGB1 as if HSL with s=1 l=0.5, then mix white/black
    r, g, b = hsl_to_rgb(h, 1, 0.5)
    rr, gg, bb = r / 255, g / 255, b / 255
    sum_wb = w + bl
    ww, bbk = w, bl
    if sum_wb > 1:
        ww = w / sum_wb
        bbk = bl / sum_wb
    
    r2 = (1 - ww - bbk) * rr + ww
    g2 = (1 - ww - bbk) * gg + ww
    b2 = (1 - ww - bbk) * bb + ww
    return int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255))

def parse_hwb(s: str) -> Optional[RGBA]:
    """Parse HWB color string to RGBA."""
    m = HWB_RE.match(s)
    if not m:
        return None
    h_val, w_val, bl_val, a_val = m.groups()
    h = 0 if h_val == "none" else angle_to_deg(h_val)
    w = 0 if w_val == "none" else clamp(pct(w_val), 0, 1)
    bl = 0 if bl_val == "none" else clamp(pct(bl_val), 0, 1)
    a = 1 if a_val is None or a_val == "none" else (
        clamp(pct(a_val), 0, 1) if a_val.endswith("%") else clamp(float(a_val), 0, 1)
    )
    r, g, b = hwb_to_rgb(h, w, bl)
    return RGBA(r=r, g=g, b=b, a=a)

# LAB/LCH (CIELAB) and OKLab/OKLCH -------------------------------

def s_to_lin(c: int) -> float:
    """sRGB companding."""
    cs = c / 255
    return cs / 12.92 if cs <= 0.04045 else ((cs + 0.055) / 1.055) ** 2.4

def lin_to_s(l: float) -> int:
    """Linear to sRGB."""
    v = 12.92 * l if l <= 0.0031308 else 1.055 * (l ** (1 / 2.4)) - 0.055
    return int(clamp(round(v * 255), 0, 255))

def srgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """sRGB D65 to XYZ."""
    R, G, B = s_to_lin(r), s_to_lin(g), s_to_lin(b)
    x = R * 0.41239079926595 + G * 0.35758433938387 + B * 0.18048078840183
    y = R * 0.21263900587151 + G * 0.71516867876775 + B * 0.07219231536073
    z = R * 0.01933081871559 + G * 0.11919477979462 + B * 0.95053215224966
    return x, y, z

def xyz_to_srgb(x: float, y: float, z: float) -> Tuple[int, int, int]:
    """XYZ to sRGB."""
    R = x * 3.24096994190452 + y * -1.53738317757009 + z * -0.498610760293
    G = x * -0.96924363628087 + y * 1.87596750150772 + z * 0.04155505740718
    B = x * 0.05563007969699 + y * -0.20397695888897 + z * 1.05697151424288
    return lin_to_s(R), lin_to_s(G), lin_to_s(B)

# LAB constants
XR, YR, ZR = 0.95047, 1.0, 1.08883

def f_lab(t: float) -> float:
    """LAB forward transform."""
    return t ** (1/3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

def f_inv_lab(t: float) -> float:
    """LAB inverse transform."""
    t3 = t * t * t
    return t3 if t3 > 216 / 24389 else (108 / 841) * (t - 4 / 29)

def lab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """Convert LAB to RGB."""
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xr = f_inv_lab(fx)
    yr = f_inv_lab(fy)
    zr = f_inv_lab(fz)
    x, y, z = xr * XR, yr * YR, zr * ZR
    return xyz_to_srgb(x, y, z)

def parse_lab(s: str) -> Optional[RGBA]:
    """Parse LAB color string to RGBA."""
    LAB_RE = re.compile(
        f"^lab{ws}\\({ws}({num}|{perc}|none)(?:{sep}|{comma})({num}|none)(?:{sep}|{comma})({num}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
        re.IGNORECASE
    )
    m = LAB_RE.match(s)
    if not m:
        return None
    L_val, a_val, b_val, alpha_val = m.groups()
    L = 0 if L_val == "none" else (
        clamp(float(L_val.replace("%", "")), 0, 100) if L_val.endswith("%") else clamp(float(L_val), 0, 100)
    )
    a = 0 if a_val == "none" else float(a_val)
    b = 0 if b_val == "none" else float(b_val)
    alpha = 1 if alpha_val is None or alpha_val == "none" else (
        clamp(pct(alpha_val), 0, 1) if alpha_val.endswith("%") else clamp(float(alpha_val), 0, 1)
    )
    r, g, b_rgb = lab_to_rgb(L, a, b)
    return RGBA(r=r, g=g, b=b_rgb, a=alpha)

def parse_lch(s: str) -> Optional[RGBA]:
    """Parse LCH color string to RGBA."""
    LCH_RE = re.compile(
        f"^lch{ws}\\({ws}({num}|{perc}|none)(?:{sep}|{comma})({num}|none)(?:{sep}|{comma})({angle}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
        re.IGNORECASE
    )
    m = LCH_RE.match(s)
    if not m:
        return None
    L_val, C_val, h_val, alpha_val = m.groups()
    L = 0 if L_val == "none" else (
        clamp(float(L_val.replace("%", "")), 0, 100) if L_val.endswith("%") else clamp(float(L_val), 0, 100)
    )
    C = 0 if C_val == "none" else float(C_val)
    h = 0 if h_val == "none" else angle_to_deg(h_val)
    # convert LCH -> LAB
    hr = (h * math.pi) / 180
    a = C * math.cos(hr)
    b = C * math.sin(hr)
    alpha = 1 if alpha_val is None or alpha_val == "none" else (
        clamp(pct(alpha_val), 0, 1) if alpha_val.endswith("%") else clamp(float(alpha_val), 0, 1)
    )
    r, g, b_rgb = lab_to_rgb(L, a, b)
    return RGBA(r=r, g=g, b=b_rgb, a=alpha)

def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """Convert OKLab to RGB."""
    # OKLab -> OKLMS -> linear sRGB -> compand
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.291485548 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    R = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    G = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    B = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s

    return lin_to_s(R), lin_to_s(G), lin_to_s(B)

def parse_oklab(s: str) -> Optional[RGBA]:
    """Parse OKLab color string to RGBA."""
    RE = re.compile(
        f"^oklab{ws}\\({ws}({num}|{perc}|none)(?:{sep}|{comma})({num}|none)(?:{sep}|{comma})({num}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
        re.IGNORECASE
    )
    m = RE.match(s)
    if not m:
        return None
    L_val, a_val, b_val, alpha_val = m.groups()
    L = 0 if L_val == "none" else (
        clamp(pct(L_val), 0, 1) if L_val.endswith("%") else clamp(float(L_val), 0, 1)
    )
    a = 0 if a_val == "none" else float(a_val)
    b = 0 if b_val == "none" else float(b_val)
    alpha = 1 if alpha_val is None or alpha_val == "none" else (
        clamp(pct(alpha_val), 0, 1) if alpha_val.endswith("%") else clamp(float(alpha_val), 0, 1)
    )
    r, g, b_rgb = oklab_to_rgb(L, a, b)
    return RGBA(r=r, g=g, b=b_rgb, a=alpha)

def parse_oklch(s: str) -> Optional[RGBA]:
    """Parse OKLCH color string to RGBA."""
    RE = re.compile(
        f"^oklch{ws}\\({ws}({num}|{perc}|none)(?:{sep}|{comma})({num}|none)(?:{sep}|{comma})({angle}|none)(?:{slash}({num}|{perc}|none))?{ws}\\)$",
        re.IGNORECASE
    )
    m = RE.match(s)
    if not m:
        return None
    L_val, C_val, h_val, alpha_val = m.groups()
    L = 0 if L_val == "none" else (
        clamp(pct(L_val), 0, 1) if L_val.endswith("%") else clamp(float(L_val), 0, 1)
    )
    C = 0 if C_val == "none" else float(C_val)
    h = 0 if h_val == "none" else angle_to_deg(h_val)
    hr = (h * math.pi) / 180
    a = C * math.cos(hr)
    b = C * math.sin(hr)
    alpha = 1 if alpha_val is None or alpha_val == "none" else (
        clamp(pct(alpha_val), 0, 1) if alpha_val.endswith("%") else clamp(float(alpha_val), 0, 1)
    )
    r, g, b_rgb = oklab_to_rgb(L, a, b)
    return RGBA(r=r, g=g, b=b_rgb, a=alpha)

# NAMED/CURRENTCOLOR ---------------------------------------------

def parse_named(s: str) -> Optional[RGBA]:
    """Parse named color string to RGBA."""
    name = s.strip().lower()
    if name == "currentcolor":
        return None  # signal special token
    hex_val = NAMED.get(name)
    if not hex_val:
        return None
    return parse_hex(hex_val)

# Top-level parse to RGBA ----------------------------------------

def parse_css_color_to_rgba(input_str: str) -> Optional[RGBA]:
    """Parse any CSS color string to RGBA."""
    s = input_str.strip()
    
    # Fast path: hex
    hex_result = parse_hex(s)
    if hex_result:
        return hex_result

    # Named and transparent
    named_result = parse_named(s)
    if named_result:
        return named_result
    if re.match(r"^transparent$", s, re.IGNORECASE):
        return RGBA(r=0, g=0, b=0, a=0)
    if re.match(r"^currentcolor$", s, re.IGNORECASE):
        return None  # leave for caller context

    # rgb/rgba
    rgb_result = parse_rgb(s)
    if rgb_result:
        return rgb_result

    # hsl/hsla
    hsl_result = parse_hsl(s)
    if hsl_result:
        return hsl_result

    # hwb
    hwb_result = parse_hwb(s)
    if hwb_result:
        return hwb_result

    # lab/lch
    lab_result = parse_lab(s)
    if lab_result:
        return lab_result

    lch_result = parse_lch(s)
    if lch_result:
        return lch_result

    # oklab/oklch
    oklab_result = parse_oklab(s)
    if oklab_result:
        return oklab_result

    oklch_result = parse_oklch(s)
    if oklch_result:
        return oklch_result

    return None

# Conversions from RGBA ------------------------------------------

def rgba_to_hex(rgba: RGBA, with_alpha: bool = True) -> str:
    """Convert RGBA to hex string."""
    def h(n: int) -> str:
        return format(n, "02x")
    
    base = f"#{h(rgba.r)}{h(rgba.g)}{h(rgba.b)}"
    if not with_alpha or rgba.a >= 1:
        return base
    return base + h(int(round(clamp(rgba.a, 0, 1) * 255)))

def rgba_to_rgb_string(rgba: RGBA) -> str:
    """Convert RGBA to RGB string."""
    if rgba.a >= 1:
        return f"rgb({rgba.r} {rgba.g} {rgba.b})"
    return f"rgb({rgba.r} {rgba.g} {rgba.b} / {rgba.a:.3f})"

def rgba_to_hsl(rgba: RGBA) -> Tuple[float, float, float]:
    """Convert RGBA to HSL."""
    R, G, B = rgba.r / 255, rgba.g / 255, rgba.b / 255
    max_val = max(R, G, B)
    min_val = min(R, G, B)
    d = max_val - min_val
    l = (max_val + min_val) / 2
    s = 0 if d == 0 else d / (1 - abs(2 * l - 1))
    h = 0
    if d != 0:
        if max_val == R:
            h = 60 * (((G - B) / d) % 6)
        elif max_val == G:
            h = 60 * ((B - R) / d + 2)
        else:
            h = 60 * ((R - G) / d + 4)
    if h < 0:
        h += 360
    return h, s, l

def rgba_to_hsl_string(rgba: RGBA) -> str:
    """Convert RGBA to HSL string."""
    h, s, l = rgba_to_hsl(rgba)
    if rgba.a >= 1:
        return f"hsl({round1(h)} {roundp(s)} {roundp(l)})"
    return f"hsl({round1(h)} {roundp(s)} {roundp(l)} / {round3(rgba.a)})"

def rgba_to_hwb_string(rgba: RGBA) -> str:
    """Convert RGBA to HWB string."""
    h, s, l = rgba_to_hsl(rgba)
    # Invert HSL to HWB:
    # Compute base RGB1 from hue
    r, g, b = hsl_to_rgb(h, 1, 0.5)
    rr, gg, bb = r / 255, g / 255, b / 255
    w = min(rgba.r / 255, rgba.g / 255, rgba.b / 255)
    bl = 1 - max(rgba.r / 255, rgba.g / 255, rgba.b / 255)
    W = clamp(w, 0, 1)
    B = clamp(bl, 0, 1)
    if rgba.a >= 1:
        return f"hwb({round1(h)} {roundp(W)} {roundp(B)})"
    return f"hwb({round1(h)} {roundp(W)} {roundp(B)} / {round3(rgba.a)})"

# XYZ helpers for LAB/OKLab conversions back from RGBA:

def rgb_to_lab(rgba: RGBA) -> Tuple[float, float, float]:
    """Convert RGB to LAB."""
    x, y, z = srgb_to_xyz(rgba.r, rgba.g, rgba.b)
    xr, yr, zr = x / XR, y / YR, z / ZR
    fx, fy, fz = f_lab(xr), f_lab(yr), f_lab(zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b

def rgba_to_lab_string(rgba: RGBA) -> str:
    """Convert RGBA to LAB string."""
    L, a, b = rgb_to_lab(rgba)
    if rgba.a >= 1:
        return f"lab({round1(L)} {round2(a)} {round2(b)})"
    return f"lab({round1(L)} {round2(a)} {round2(b)} / {round3(rgba.a)})"

def rgba_to_lch_string(rgba: RGBA) -> str:
    """Convert RGBA to LCH string."""
    L, a, b = rgb_to_lab(rgba)
    C = math.sqrt(a * a + b * b)
    h = (math.atan2(b, a) * 180) / math.pi
    if h < 0:
        h += 360
    if rgba.a >= 1:
        return f"lch({round1(L)} {round2(C)} {round1(h)})"
    return f"lch({round1(L)} {round2(C)} {round1(h)} / {round3(rgba.a)})"

def rgb_to_oklab(rgba: RGBA) -> Tuple[float, float, float]:
    """Convert RGB to OKLab."""
    # sRGB -> linear
    R, G, B = s_to_lin(rgba.r), s_to_lin(rgba.g), s_to_lin(rgba.b)

    l = (0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B) ** (1/3)
    m = (0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B) ** (1/3)
    s = (0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B) ** (1/3)

    L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s
    b = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s

    return L, a, b

def rgba_to_oklab_string(rgba: RGBA) -> str:
    """Convert RGBA to OKLab string."""
    L, a, b = rgb_to_oklab(rgba)
    if rgba.a >= 1:
        return f"oklab({round3(L)} {round3(a)} {round3(b)})"
    return f"oklab({round3(L)} {round3(a)} {round3(b)} / {round3(rgba.a)})"

def rgba_to_oklch_string(rgba: RGBA) -> str:
    """Convert RGBA to OKLCH string."""
    L, a, b = rgb_to_oklab(rgba)
    C = math.sqrt(a * a + b * b)
    h = (math.atan2(b, a) * 180) / math.pi
    if h < 0:
        h += 360
    if rgba.a >= 1:
        return f"oklch({round3(L)} {round3(C)} {round1(h)})"
    return f"oklch({round3(L)} {round3(C)} {round1(h)} / {round3(rgba.a)})"

def rgba_to_named(rgba: RGBA) -> Optional[str]:
    """Convert RGBA to named color if exact match exists."""
    hex_val = rgba_to_hex(rgba, rgba.a < 1)
    # exact match only
    for name, h in NAMED.items():
        if h.lower() == hex_val.lower():
            return name
    return None

# Targeted conversion API ----------------------------------------

router = APIRouter()

@router.post("/convert_color_code", response_model=SuccessResponse, operation_id="convert_color_code", description="Convert a CSS color code to a target format")
async def parse_and_convert(request: ColorConvertRequest):
    """Parse CSS color and convert to target format."""
    input_str = request.code
    target = request.target
    rgba = parse_css_color_to_rgba(input_str)
    if not rgba:
        raise HTTPException(status_code=400, detail="Invalid CSS color or currentColor")

    if target == "hex":
        return SuccessResponse(success=True, message=rgba_to_hex(rgba, True))
    elif target == "rgb":
        return SuccessResponse(success=True, message=rgba_to_rgb_string(rgba))
    elif target == "hsl":
        return SuccessResponse(success=True, message=rgba_to_hsl_string(rgba))
    elif target == "hwb":
        return SuccessResponse(success=True, message=rgba_to_hwb_string(rgba))
    elif target == "lab":
        return SuccessResponse(success=True, message=rgba_to_lab_string(rgba))
    elif target == "lch":
        return SuccessResponse(success=True, message=rgba_to_lch_string(rgba))
    elif target == "oklab":
        return SuccessResponse(success=True, message=rgba_to_oklab_string(rgba))
    elif target == "oklch":
        return SuccessResponse(success=True, message=rgba_to_oklch_string(rgba))
    elif target == "named":
        return SuccessResponse(success=True, message=rgba_to_named(rgba))
    else:
        raise HTTPException(status_code=400, detail="Invalid target color space")

# Formatting helpers ---------------------------------------------

def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x * 10) / 10

def round2(x: float) -> float:
    """Round to 2 decimal places."""
    return round(x * 100) / 100

def round3(x: float) -> float:
    """Round to 3 decimal places."""
    return round(x * 1000) / 1000

def roundp(x: float) -> str:
    """For percentages in CSS strings: keep 1 decimal for nicer output."""
    v = clamp(x, 0, 1)
    return f"{round(v * 1000) / 10}%"

# Pydantic validation wrapper
class CssColorString(str):
    """Pydantic validator for CSS color strings."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError('string required')
        if parse_css_color_to_rgba(v) is None:
            raise ValueError('Invalid CSS color or currentColor')
        return cls(v)
