# ComfyUI: PNG → Color/Line Split + JSX

把一张 PNG 自动拆分成 **颜色层** 与 **线稿层** 两张图，并导出一个 **Photoshop JSX**，
在 PS 里执行即可生成 **带图层的 PSD**（上：Lineart，下：Color）。

---

## ✨ 功能概述
- 输入：`IMAGE`（单张或批量）  
- 输出：两路 `IMAGE`（颜色层、线稿层）  
- 自动落地：
  ```
  outputs/png2psd/YYYYMMDD_HHMMSS/
  ├─ color.png
  ├─ line.png
  └─ make_psd.jsx
  ```
- JSX 使用 Photoshop 原生脚本生成分层 PSD（兼容性最强）。

## 🧩 节点参数说明

| 参数 | 类型 | 说明 | 推荐范围 |
|------|------|------|-----------|
| `ink_L_black_threshold` | int | 线条黑度阈值(L*)，越大越宽松，可抓到更浅灰线 | 55–70 |
| `desaturate_threshold` | int | 去彩度阈值，越小越“灰黑”，避免彩色阴影被当作线 | 12–22 |
| `line_thicken_px` | int | 线条加粗像素，用于修补细线缺口 | 1–3 |
| `inpaint_radius` | int | 颜色修复半径，去掉颜色层残影 | 3–6 |
| `output_parent_dir` | str | 输出父目录，节点会自动在里面建时间戳文件夹 | outputs/png2psd |

---

## 🖼️ 示例工作流
见 `comfyui-png2psd-example.json`。

### 导入方法：
1. 打开 ComfyUI  
2. 点击 **Load → 选择 comfyui-png2psd-example.json**  
3. 在 `Load Image` 节点中选一张 PNG  
4. 点击 **Queue Prompt** 运行

运行后会：
- 输出两张预览图（颜色层、线稿层）  
- 自动生成：
  ```
  outputs/png2psd/YYYYMMDD_HHMMSS/
  ├─ color.png
  ├─ line.png
  └─ make_psd.jsx
  ```

---

## 🧾 许可证
MIT License

---

## 👤 作者
**Mike-Dai**
