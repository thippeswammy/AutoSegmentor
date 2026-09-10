# Assets

Media referenced by the top-level [`README.md`](../README.md) and the docs site: poster
thumbnails and full-quality showcase recordings of the app running. Demo *input* footage
(what the pipeline actually processes) lives in
[`demo/videos/`](../demo/videos/README.md) instead — generated datasets, masks, and
working files are never committed anywhere.

## Contents

| File | Used in | Description |
| :--- | :--- | :--- |
| `AutoSegmenterCat.mp4` | README hero, `docs/index.md`, `docs/demos.md` | Full-quality, full-length recording of a real `cat` demo annotation session — UI annotation, SAM2 + CoTracker3 auto-tracking, correction, export. 1920×1080, 30fps, ~58MB. |
| `AutoSegmenterRoad.mp4` | README "Demos" section, `docs/demos.md` | Recording of the `road` demo (SAM2 segmentation only, no pose tracking). Byte-identical to `demo/videos/road_dashboard.mp4` — the road demo's raw dashcam footage doubles as its own showcase video. ~88MB. |
| `cat_poster.jpg` | `<video poster>` for the cat recording | Representative frame (mid-annotation, CoTracker keypoints visible) shown before playback starts. |
| `road_poster.jpg` | `<video poster>` for the road recording | Representative frame (segmentation masks visible) shown before playback starts. |

## Embedding these videos — the LFS gotcha

Both `.mp4` files are tracked via Git LFS. **`raw.githubusercontent.com` does not resolve
LFS content** — it serves the tiny LFS pointer text file instead of the real video, which
silently breaks `<video>` playback (the player renders, nothing plays). The fix is GitHub's
dedicated LFS media endpoint instead:

```html
<video controls preload="metadata" poster="https://raw.githubusercontent.com/<owner>/<repo>/<branch>/assets/cat_poster.jpg">
  <source src="https://media.githubusercontent.com/media/<owner>/<repo>/<branch>/assets/AutoSegmenterCat.mp4" type="video/mp4">
</video>
```

- Poster images are small plain files (not LFS-tracked), so `raw.githubusercontent.com`
  serves those correctly — only the LFS-tracked `.mp4`/`.gif` need the `media.` host.
- `preload="metadata"` avoids downloading the full 58–88MB file just because the page
  loaded; the browser only fetches enough to show duration/dimensions until the visitor
  presses play.
- No `autoplay` — combined with the file sizes here, autoplay would force a full download
  on page load for every visitor, which is exactly the latency this setup avoids.

## Guidelines for adding media here

- **Git LFS is required** for new media added to this repo: run `git lfs install`
  once, then `.gitattributes` (repo root) automatically routes any new `.mp4`/`.gif`
  through LFS instead of a plain git blob.
- Any new video embedded via `<video>`/`<source>` **must** use the
  `media.githubusercontent.com/media/...` host, not `raw.githubusercontent.com` — see above.
- Committing a video here via LFS is fine for files in the low hundreds of MB. For anything
  approaching GitHub's free LFS quota (1GB storage / 1GB bandwidth per month), prefer a
  [GitHub Release](https://github.com/thippeswammy/AutoSegmentor/releases) asset (up to 2GB,
  doesn't count against LFS quota) instead.
- `demo/videos/*.mp4` are also LFS-tracked (migrated in a later commit, no history rewrite)
  — they're the pipeline's raw inputs, not showcase assets, so they stay in `demo/videos/`
  rather than here.
