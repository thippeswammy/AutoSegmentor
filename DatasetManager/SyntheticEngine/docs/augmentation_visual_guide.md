# 🧠 SyntheticEngine — Augmentation Visual Reference Guide

> **Config file:** `DatasetManager/SyntheticEngine/config/default_config.yaml`  
> **Use case:** Pallet detection (warehouse, top/side views)

---

## 1. 🔄 GEOMETRIC Augmentations

> Controls **how the object's shape and orientation** changes.  
> `prob: 0.2` → only 20% of images will go through this stage at all.

![Rotation and Flip](geometric_rotation_flip_1779186184798.png)

![Scale, Perspective and Shear](geometric_scale_perspective_shear_1779186234122.png)

| Parameter | What It Does to the Image | Your Current Value | Recommended for Pallets |
|---|---|---|---|
| `prob` | Chance this whole geometric stage runs | `0.2` (20%) | `0.5–0.7` if you want more variety |
| `rotate_limit` | Random rotation ±N degrees. `0` = no rotation | `0` (disabled) | `0–15°` for realistic camera angles |
| `h_flip_prob` | % chance to mirror image left↔right | `0.0` (off) | `0.5` — pallets look same flipped |
| `v_flip_prob` | % chance to flip image top↔bottom | `0.0` (off) | `0.0` — upside-down pallet is unrealistic |
| `scale_range` | Resize object between min and max factor before pasting | `[1, 1]` (no change) | `[0.4, 1.2]` to simulate distance |
| `perspective_scale` | Strength of perspective warp (tilt/angle distortion) | `[0.01, 0.1]` | `[0.01, 0.05]` subtle tilt is realistic |
| `shear_limit` | Max degrees of horizontal shear/skew | `4` | `2–5` is fine for slight camera lean |

> [!TIP]
> For pallets, `h_flip: 0.5` is safe (they're symmetric). Keep `v_flip: 0.0` — upside-down pallets don't exist in real warehouses.

---

## 2. 📋 COPY-PASTE Augmentations

> Controls **how the object (pallet cutout) is pasted onto a background image.**

![Copy-Paste Scale and Blending](copy_paste_scale_blend_1779186476885.png)

| Parameter | What It Does to the Image | Your Current Value | Recommended |
|---|---|---|---|
| `prob` | Average number of objects pasted per background | `6.0` | `1.0–4.0` — 6 may crowd the image |
| `object_scale_range` | Min/max size of pasted object (simulates distance) | `[0.3, 1.0]` | ✅ Good — covers near & far pallets |
| `alpha_blend_sigma` | How soft/feathered the paste edge is. Low=sharp, High=blurry edge | `[3, 8]` | ✅ Good — range gives variety |
| `histogram_match` | Adjusts object colors to match background lighting | `false` | `true` for more realistic compositing |

---

### 2a. 🔀 Inversion Effects

![Inversion and Histogram Match](copy_paste_inversion_histogram_1779186496153.png)

> Inversion = **color negative** effect. Like a photo negative. Makes the model robust to unusual sensor readings or thermal cameras.

| Parameter | What It Does to the Image | Your Current Value | Recommended |
|---|---|---|---|
| `enabled` | Master toggle for all inversion | `true` | ✅ Keep `true` |
| `object_only_prob` | Chance ONLY the pallet is inverted, background is normal | `0.10` (10%) | `0.05–0.15` |
| `full_image_prob` | Chance ENTIRE image (object + background) is inverted | `0.05` (5%) | `0.02–0.05` — use rarely |
| `intensity_range` | How strong the inversion is. `0.5` = partial, `1.0` = full negative | `[0.5, 1.0]` | ✅ Good range |

> [!NOTE]
> `object_only_prob` = only the pallet looks negative (alien-like). `full_image_prob` = the entire scene looks inverted. Both help if you have unusual IR or thermal camera data.

---

### 2b. 💡 Lighting Effects (Shadow & Glare)

> **No generated image available** (quota exhausted), but here's a clear text description:

| Effect | What You See | Parameter | Your Value |
|---|---|---|---|
| **Shadow (multiply)** | Pallet gets darker, like it's in shadow | `multiply_prob: 0.30`, `multiply_range: [0.50, 0.85]` | `0.50` = very dark shadow, `0.85` = light shadow |
| **Glare (add)** | Pallet gets brighter/washed out, like direct light hit | `add_prob: 0.20`, `add_range: [20, 70]` | `+20` = mild glare, `+70` = near overexposed |

```
multiply x 0.50  →  [very dark pallet, deep shadow]
multiply x 0.85  →  [slightly dimmed pallet, soft shadow]
add + 20         →  [slightly brighter, mild highlight]
add + 70         →  [very bright, blown out glare]
```

> [!TIP]
> `multiply_range [0.50, 0.85]` is a good range. If pallets are always in bright warehouse light, you can narrow to `[0.70, 0.95]` for subtler shadows.

---

## 3. 🎨 PHOTOMETRIC Augmentations

> Controls **how the whole image's color, sharpness, and quality changes.**  
> `prob: 0.8` → 80% of images go through this stage.

![Brightness, Contrast and Hue](photometric_brightness_contrast_hue_1779186571761.png)

| Parameter | What You See | Your Current Value | Recommended |
|---|---|---|---|
| `brightness_range` | `[-0.1, 0.4]` = can go slightly dark OR noticeably brighter | `[-0.1, 0.4]` | ✅ Slight negative okay, big positive helps with glare |
| `contrast_range` | `0.6` = flat washed-out look. `1.4` = punchy deep shadows | `[0.60, 1.40]` | ✅ Good range covers dim and harsh lighting |
| `hue_shift_limit` | Slight color tone shift (warm/cool). `±8` is subtle | `8` | ✅ Good — slight tint variation |
| `saturation_range` | `0.4` = almost grayscale. `0.9` = near full color | `[0.4, 0.9]` | ✅ Helps with grey/dusty warehouse floors |

### Degradation Effects (Blur / Noise / JPEG)

> These simulate **bad camera conditions** — motion blur, cheap sensors, video compression.

| Parameter | What You See | Your Current Value | Notes |
|---|---|---|---|
| `blur_prob` | Gaussian/motion blur — like camera shake or fast movement | `0.10` (10%) | ✅ 10% is good — don't overdo it |
| `noise_prob` | Random pixel grain — like low-light/high-ISO camera | `0.05` (5%) | ✅ 5% is realistic |
| `jpeg_prob` | JPEG compression blocks/artifacts — like low-quality video frame | `0.10` (10%) | ✅ Important for CCTV/IP camera footage |

> [!TIP]
> If your real deployment uses **IP cameras or video streams**, increase `jpeg_prob` to `0.15–0.25` since video frames always have compression artifacts.

---

## 4. ⬛ OCCLUSION Augmentations

> Randomly places **black/dark rectangles OVER the pallet** to simulate objects blocking the view.  
> `prob: 0.3` → 30% of images will have occlusion patches.

```
ORIGINAL:           [full pallet visible]  ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅
1 patch (5%):       [pallet with tiny ■ blocking one corner]
2 patches (15%):    [pallet with two ■ ■ blocking sides]
3 patches (30%):    [pallet with three large ■ ■ ■ — heavily blocked]
```

| Parameter | What It Does | Your Current Value | Recommended |
|---|---|---|---|
| `prob` | Chance this occlusion stage applies | `0.3` (30%) | ✅ Good for teaching partial visibility |
| `max_patches` | Max number of blocking rectangles drawn | `3` | `2–4` is realistic |
| `patch_size_min` | Smallest patch = 5% of pallet bounding box | `0.05` | ✅ Small patches (like a bolt/pipe) |
| `patch_size_max` | Largest patch = 30% of pallet bounding box | `0.30` | ✅ Large patches (like another pallet in front) |

> [!IMPORTANT]
> Occlusion is **very important for warehouse pallets** — forklifts, other pallets, or boxes are constantly partially blocking them. Consider increasing `prob` to `0.4–0.5`.

---

## 📊 Quick Summary: Which Parameters Matter Most for Pallets?

| Priority | Parameter | Why |
|---|---|---|
| 🔴 High | `copy_paste.object_scale_range` | Pallet distance variation is critical |
| 🔴 High | `photometric.brightness_range` | Warehouse lighting is very variable |
| 🔴 High | `occlusion.prob` | Pallets are often partially hidden |
| 🟡 Medium | `geometric.h_flip_prob` | Pallets look same when flipped |
| 🟡 Medium | `photometric.jpeg_prob` | CCTV/IP cameras compress heavily |
| 🟡 Medium | `copy_paste.histogram_match` | Enable for more realistic compositing |
| 🟢 Low | `geometric.v_flip_prob` | Keep at 0 — upside-down = unrealistic |
| 🟢 Low | `copy_paste.inversion.full_image_prob` | Keep low, rarely useful |
