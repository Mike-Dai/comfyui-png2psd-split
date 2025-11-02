# -*- coding: utf-8 -*-
# ComfyUI Custom Node: PNG -> (Color/Line) split + JSX exporter
# Repository: comfyui-png2psd-split
# Description:
#   - 输入一张/多张 IMAGE，输出两张 IMAGE（颜色层、线稿层）
#   - 自动在“输出父目录/时间戳/”生成 color.png / line.png / make_psd.jsx
#   - JSX 在 Photoshop 里执行即可合成分层 PSD（上：Lineart，下：Color）

from datetime import datetime
from pathlib import Path

import numpy as np
import cv2 as cv
from PIL import Image
import torch

# ---------------- 核心算法：LAB 近黑线稿 + Inpaint 颜色 ----------------
def extract_layers_rgba(
    rgba: np.ndarray,
    ink_l: int = 60,
    ink_chroma: int = 18,
    dilate_px: int = 2,
    inpaint_r: int = 4,
):
    """
    rgba: HxWx4 uint8
    return: color_rgba, line_rgba (uint8 RGBA)
    """
    rgb = cv.cvtColor(rgba, cv.COLOR_RGBA2RGB)
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
    L, A, B = cv.split(lab)

    A = A.astype(np.int16) - 128
    B = B.astype(np.int16) - 128
    chroma = np.sqrt(A * A + B * B).astype(np.float32)

    alpha = rgba[..., 3]
    # “近黑 + 低彩度 + 有效 alpha” 视为线
    ink_mask = (L < ink_l) & (chroma < ink_chroma) & (alpha > 0)
    ink_mask = (ink_mask.astype(np.uint8) * 255)

    # 修补小断裂 + 轻度膨胀（实心线，更利于 inpaint）
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    ink_mask = cv.morphologyEx(ink_mask, cv.MORPH_CLOSE, k, iterations=1)
    if dilate_px > 0:
        k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        ink_mask = cv.dilate(ink_mask, k2, iterations=1)

    H, W = ink_mask.shape
    line_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    line_rgba = np.dstack([line_rgb, ink_mask])

    # 颜色层：对线区域做 inpaint，去除黑边
    mask = ink_mask
    base_rgb = rgb.copy()
    if inpaint_r > 0 and mask.any():
        base_rgb = cv.inpaint(base_rgb, mask, inpaint_r, cv.INPAINT_TELEA)
    color_rgba = np.dstack([base_rgb, alpha])

    return color_rgba.astype(np.uint8), line_rgba.astype(np.uint8)


def write_jsx_copy_paste(color_png: Path, line_png: Path, out_psd: Path, w: int, h: int, jsx_path: Path):
    """生成 Photoshop JSX（复制/粘贴方案，兼容性最好）"""
    color_png = color_png.resolve().as_posix()
    line_png  = line_png.resolve().as_posix()
    out_psd   = out_psd.resolve().as_posix()

    jsx = f"""#target photoshop
app.displayDialogs = DialogModes.NO;

function openDoc(p) {{ return app.open(new File(p)); }}
function newTarget(w, h, res) {{
  return app.documents.add(w, h, res, "AutoSplit", NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
}}
function copyPasteToTarget(srcDoc, dstDoc) {{
  app.activeDocument = srcDoc;
  srcDoc.selection.selectAll();
  srcDoc.selection.copy(true);
  app.activeDocument = dstDoc;
  var layer = dstDoc.paste();
  return layer;
}}

var colorDoc = openDoc("{color_png}");
var target = newTarget({w}, {h}, 72);
var colorLayer = copyPasteToTarget(colorDoc, target);
colorLayer.name = "Color";
colorDoc.close(SaveOptions.DONOTSAVECHANGES);

var lineDoc = openDoc("{line_png}");
var lineLayer = copyPasteToTarget(lineDoc, target);
lineLayer.name = "Lineart";
lineLayer.move(target.layers[0], ElementPlacement.PLACEBEFORE);
lineDoc.close(SaveOptions.DONOTSAVECHANGES);

var outFile = new File("{out_psd}");
var opts = new PhotoshopSaveOptions();
opts.layers = true; opts.embedColorProfile = true;
target.saveAs(outFile, opts, true);
target.close(SaveOptions.DONOTSAVECHANGES);
"""
    jsx_path.write_text(jsx, encoding="utf-8")


# ---------------- ComfyUI 张量 <-> RGBA ----------------
def tensor_to_rgba_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [H,W,C] or [1,H,W,C], float32 in [0,1], C=3/4
    return: HxWx4 uint8
    """
    if img_t.ndim == 4:
        img_t = img_t[0]
    img = img_t.detach().cpu().numpy()
    if img.shape[2] == 3:  # 无 alpha -> 补全
        a = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, a], axis=2)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def rgba_uint8_to_tensor(rgba: np.ndarray) -> torch.Tensor:
    """HxWx4 uint8 -> [H,W,4] float32 [0,1]"""
    arr = rgba.astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ---------------- ComfyUI 自定义节点 ----------------
class PNG2PSD_SplitNode:
    """
    输入 IMAGE，输出 颜色层 与 线稿层（IMAGE），并落地 PNG 与 JSX。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ink_L_black_threshold": ("INT", {"default": 60, "min": 30, "max": 100, "step": 1}),
                "desaturate_threshold":  ("INT", {"default": 18, "min": 5,  "max": 40,  "step": 1}),
                "line_thicken_px":      ("INT", {"default": 2,  "min": 0,  "max": 6,   "step": 1}),
                "inpaint_radius":       ("INT", {"default": 4,  "min": 0,  "max": 8,   "step": 1}),
                "output_parent_dir":    ("STRING", {"default": "outputs/png2psd"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("color_image", "line_image")
    FUNCTION = "run"
    CATEGORY = "Utils/PNG→PSD"

    def run(self, image,
            ink_L_black_threshold, desaturate_threshold,
            line_thicken_px, inpaint_radius,
            output_parent_dir):

        parent = Path(output_parent_dir)
        parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = parent / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        # 统一 batch 维
        if image.ndim == 4:    # [B,H,W,C]
            batch = image
        else:                  # [H,W,C] -> [1,H,W,C]
            batch = image.unsqueeze(0)

        color_list, line_list = [], []

        for idx in range(batch.shape[0]):
            rgba = tensor_to_rgba_uint8(batch[idx])

            color_rgba, line_rgba = extract_layers_rgba(
                rgba,
                ink_l=ink_L_black_threshold,
                ink_chroma=desaturate_threshold,
                dilate_px=line_thicken_px,
                inpaint_r=inpaint_radius,
            )

            suffix = "" if batch.shape[0] == 1 else f"_{idx:02d}"
            color_png = out_dir / f"color{suffix}.png"
            line_png  = out_dir / f"line{suffix}.png"
            psd_path  = out_dir / f"layered{suffix}.psd"
            jsx_path  = out_dir / f"make_psd{suffix}.jsx"

            Image.fromarray(color_rgba, "RGBA").save(color_png.as_posix())
            Image.fromarray(line_rgba,  "RGBA").save(line_png.as_posix())

            h, w = rgba.shape[:2]
            write_jsx_copy_paste(color_png, line_png, psd_path, w, h, jsx_path)

            color_list.append(rgba_uint8_to_tensor(color_rgba))
            line_list.append(rgba_uint8_to_tensor(line_rgba))

        color_out = torch.stack(color_list, dim=0)
        line_out  = torch.stack(line_list,  dim=0)
        return (color_out, line_out)


NODE_CLASS_MAPPINGS = {"PNG2PSD_SplitNode": PNG2PSD_SplitNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PNG2PSD_SplitNode": "PNG → Color/Line Split + JSX"}
