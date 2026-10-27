"""用 Pillow 绘制报告插图所需的简易绘图工具（折线图、热力图、图片网格）。"""
import os

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
_CACHE = {}

INK = (40, 40, 40)
GRID = (225, 225, 225)
AXIS = (120, 120, 120)


def font(size, bold=False):
    key = (size, bold)
    if key not in _CACHE:
        name = "msyhbd.ttc" if bold else "msyh.ttc"
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            path = os.path.join(FONT_DIR, "simhei.ttf")
        _CACHE[key] = ImageFont.truetype(path, size)
    return _CACHE[key]


def canvas(width, height, bg=(255, 255, 255)):
    img = Image.new("RGB", (width, height), bg)
    return img, ImageDraw.Draw(img)


def text(draw, xy, s, size=22, bold=False, fill=INK, anchor="la"):
    draw.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def axes(draw, rect, xlim, ylim, xticks=None, yticks=None, xlabel="", ylabel="",
         tick_size=18, label_size=20, y_fmt="{:.2f}", x_fmt="{}", grid=True):
    """rect=(left, top, right, bottom) 像素坐标；返回坐标映射函数。"""
    l, t, r, b = rect
    x0, x1 = xlim
    y0, y1 = ylim

    def fx(v):
        return l + (v - x0) / (x1 - x0) * (r - l)

    def fy(v):
        return b - (v - y0) / (y1 - y0) * (b - t)

    if grid and yticks:
        for yv in yticks:
            y = fy(yv)
            draw.line([l - 6, y, r, y], fill=GRID, width=2)
    draw.line([l, b, r, b], fill=AXIS, width=2)
    draw.line([l, t, l, b], fill=AXIS, width=2)
    if yticks:
        for yv in yticks:
            y = fy(yv)
            draw.line([l - 6, y, l, y], fill=AXIS, width=2)
            text(draw, (l - 12, y), y_fmt.format(yv), size=tick_size, anchor="rm")
    if xticks:
        for xv in xticks:
            x = fx(xv)
            draw.line([x, b, x, b + 6], fill=AXIS, width=2)
            text(draw, (x, b + 10), x_fmt.format(xv), size=tick_size, anchor="ma")
    if xlabel:
        text(draw, ((l + r) / 2, b + 42), xlabel, size=label_size, anchor="ma")
    if ylabel:
        text(draw, (l - 70, (t + b) / 2), ylabel, size=label_size, anchor="mm")
    return fx, fy


def polyline(draw, pts, color, width=4):
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")


def markers(draw, pts, color, r=6):
    for x, y in pts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=2)


def legend(draw, items, xy, size=18):
    """items: [(label, color)]"""
    x, y = xy
    for i, (label, color) in enumerate(items):
        yy = y + i * 34
        draw.line([x, yy, x + 34, yy], fill=color, width=6)
        text(draw, (x + 44, yy), label, size=size, anchor="lm")


def save(img, path, dpi_note=None):
    img.save(path, "PNG")
    return path