"""Compare CoTracker vs LK optical flow pose tracking results."""
import json
import math

CT_PATH = "outputs/pose_dataset_co_tracker/pose_labels.json"
LK_PATH = "outputs/pose_dataset_opticalflow/pose_labels.json"

ct = json.load(open(CT_PATH))
lk = json.load(open(LK_PATH))

print(f"CoTracker frames: {len(ct)}, LK frames: {len(lk)}")
print()

kp_names = [kp["name"] for kp in ct[0]["keypoints"]]
n_kps = len(kp_names)
n_frames = min(len(ct), len(lk))

# Header
header = f"{'Keypoint':<22} {'Avg Dist':>10} {'Max Dist':>10} {'CT Vis%':>8} {'LK Vis%':>8}"
print(header)
print("-" * len(header))

total_dist = 0
total_count = 0

for ki in range(n_kps):
    dists = []
    ct_vis = 0
    lk_vis = 0
    for fi in range(n_frames):
        ck = ct[fi]["keypoints"][ki]
        lkk = lk[fi]["keypoints"][ki]
        ct_vis += 1 if ck.get("visible", 0) > 0 else 0
        lk_vis += 1 if lkk.get("visible", 0) > 0 else 0
        if ck.get("visible", 0) > 0 and lkk.get("visible", 0) > 0:
            d = math.sqrt((ck["x"] - lkk["x"]) ** 2 + (ck["y"] - lkk["y"]) ** 2)
            dists.append(d)
    avg_d = sum(dists) / len(dists) if dists else 0
    max_d = max(dists) if dists else 0
    total_dist += sum(dists)
    total_count += len(dists)
    ct_pct = f"{100 * ct_vis / n_frames:.1f}%"
    lk_pct = f"{100 * lk_vis / n_frames:.1f}%"
    print(f"{kp_names[ki]:<22} {avg_d:>10.2f} {max_d:>10.2f} {ct_pct:>8} {lk_pct:>8}")

print("-" * len(header))
overall = total_dist / total_count if total_count else 0
print(f"Overall avg pixel distance: {overall:.2f}")
print()

# Frame-by-frame divergence (sampled)
print("Divergence over time (sampled every 50 frames):")
print(f"{'Frame':>6} {'Avg Dist':>10}")
for fi in range(0, n_frames, 50):
    d_sum = 0
    d_cnt = 0
    for ki in range(n_kps):
        ck = ct[fi]["keypoints"][ki]
        lkk = lk[fi]["keypoints"][ki]
        if ck.get("visible", 0) > 0 and lkk.get("visible", 0) > 0:
            d = math.sqrt((ck["x"] - lkk["x"]) ** 2 + (ck["y"] - lkk["y"]) ** 2)
            d_sum += d
            d_cnt += 1
    avg = d_sum / d_cnt if d_cnt else 0
    print(f"{fi:>6} {avg:>10.2f}")

# Also show last frame
fi = n_frames - 1
d_sum = 0
d_cnt = 0
for ki in range(n_kps):
    ck = ct[fi]["keypoints"][ki]
    lkk = lk[fi]["keypoints"][ki]
    if ck.get("visible", 0) > 0 and lkk.get("visible", 0) > 0:
        d = math.sqrt((ck["x"] - lkk["x"]) ** 2 + (ck["y"] - lkk["y"]) ** 2)
        d_sum += d
        d_cnt += 1
avg = d_sum / d_cnt if d_cnt else 0
print(f"{fi:>6} {avg:>10.2f}  (last frame)")
