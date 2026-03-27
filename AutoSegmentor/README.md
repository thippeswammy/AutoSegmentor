# AutoSegmentor

A model-agnostic video segmentation and pose estimation pipeline that integrates **SAM2** (for mask generation) and **CoTracker / Lucas-Kanade** (for keypoint tracking) via an interactive annotation UI.

---

## 🚀 Quick Start

```powershell
python autosegmentor_demo.py
```

All parameters are controlled by `inputs/config/default_config.yaml`. No code changes needed for most workflows.

---

## ⚙️ Configuration

Edit `inputs/config/default_config.yaml`:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `video_start` / `video_end` | `int` | Range of videos to process. |
| `prefix` | `str` | Filename prefix (e.g., `"Img"`). |
| `batch_size` | `int` | Frames per GPU pass — tune for your VRAM. |
| `fps` | `int` | Output video frame rate. |
| `delete` | `bool` | Auto-delete temp files after processing. |
| `run_mode` | `str` | `"all"` (SAM2 + CoTracker), `"sam_only"`, or `"pose_only"`. |
| `auto_prompt_encoding` | `bool` | Propagate masks from previous frames automatically. |

**SAM2 section:**
```yaml
sam:
  enabled: true
```

**Pose estimation section:**
```yaml
pose_estimation:
  enabled: true
  cotracker:
    checkpoint: "../co-tracker/checkpoints/scaled_offline.pth"
    window_len: 60
```

---

## 🎮 Annotation GUI Controls

| Key / Action | Effect |
| :--- | :--- |
| Left Click | Add foreground point (positive prompt) |
| Right Click | Add background point (negative prompt) |
| `Tab` | Next object instance ID |
| `1`–`9` | Select class label |
| `Enter` | Confirm and continue processing |
| `Q` | Quit annotator |

> **Tip:** Use the floating **Zoom View** window to place precise points on small objects.

---

## 📂 Directory Layout

```
AutoSegmentor/
├── autosegmentor_demo.py          ← Main entry point
├── inputs/
│   ├── config/default_config.yaml ← All pipeline settings
│   ├── VideoInputs/               ← Place raw .mp4 files here (gitignored)
│   └── UserPrompts/               ← Saved annotation JSON files
├── outputs/                       ← Final videos (gitignored)
├── working_dir/                   ← Temp processing dir (gitignored)
└── utils/
    ├── Core/
    │   └── AutoSegmentorEngine.py ← Main orchestration engine
    ├── Models/
    │   ├── SAM/
    │   │   ├── AppConfig.py       ← Pipeline configuration class
    │   │   └── SAM2Model.py       ← SAM2 model loader
    │   └── Tracking/
    │       ├── CoTrackerPredictor.py  ← CoTracker integration
    │       └── LKKeypointTracker.py  ← Lucas-Kanade fallback tracker
    ├── FileManagement/            ← Frame I/O, mask processing, export
    └── UserUI/                    ← PyQt5 annotation interface
```

---

## 🧪 Running Tests

```powershell
# Unit + integration tests (no GPU needed)
python -m pytest tests/unit tests/integration -v

# UI tests (requires display)
python -m pytest tests/ui -v
```

---

## 📦 Architecture

```
autosegmentor_demo.py
        │
        ▼
   pipeline.py  ──────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
AutoSegmentorEngine                               PoseExporter
  ├─ SAM2Model (mask generation)              ├─ CoTrackerPredictor
  ├─ CoTrackerPredictor (pose tracking)       └─ LKKeypointTracker
  └─ UserInteractionHandler (Qt UI)
```

For detailed utility documentation see [`utils/README.md`](utils/README.md).
