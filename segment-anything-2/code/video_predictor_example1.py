import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Device selection and setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_context = nullcontext()

# Load SAM 2 predictor
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Helper functions


# Initialization and state setup
video_dir = "../videos/Temp"
frame_names = sorted(
    [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# Example segmentation points
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
labels = np.array([1, 1], np.int32)
ann_frame_idx = 0
ann_obj_id = 1

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Run propagation and save frames
video_segments = {}
rendered_dir = "../rendered_frames"
os.makedirs(rendered_dir, exist_ok=True)

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# plt.close("all")

with autocast_context:
    for out_frame_idx in range(0, len(frame_names)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}")
        ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        ax.axis('off')
        fig.tight_layout()
        fig.gca().set_position([0, 0, 1, 1])

        # Save the figure as an image using PIL
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image.save(os.path.join(rendered_dir, f"rendered_frame_{out_frame_idx:05d}.png"))

        # plt.close(fig)

# import os
# from contextlib import nullcontext
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from PIL import Image
#
# # Device selection and setup
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
#
# if device.type == "cuda":
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# else:
#     autocast_context = nullcontext()
#
# # Load SAM 2 predictor
# from sam2.build_sam import build_sam2_video_predictor
#
# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
#
#
# # Helper functions
# def show_mask(mask, ax, obj_id=None, random_color=False):
#     # Convert mask to numpy array if it's a tensor
#     if torch.is_tensor(mask):
#         mask = mask.cpu().numpy()
#
#     # Generate a random color if specified or use a fixed color
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([*plt.get_cmap("tab10")(obj_id if obj_id is not None else 0)[:3], 0.6])
#
#     # Convert the mask to a 2D array if necessary
#     mask_shape = mask.shape[-2:]
#     mask_2d = mask.reshape(mask_shape)
#
#     # Make sure the color array is properly reshaped
#     color_reshaped = color.reshape(1, 1, -1)
#
#     # Apply the color to the mask
#     ax.imshow(mask_2d[..., np.newaxis] * color_reshaped)
#
#
# # Initialization and state setup
# video_dir = "videos/bedroom"
# frame_names = sorted(
#     [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#     key=lambda p: int(os.path.splitext(p)[0])
# )
# inference_state = predictor.init_state(video_path=video_dir)
# predictor.reset_state(inference_state)
#
# # Example segmentation points
# points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# labels = np.array([1, 1], np.int32)
# ann_frame_idx = 0
# ann_obj_id = 1
#
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )
#
# # Run propagation and save frames
# video_segments = {}
# rendered_dir = "./rendered_frames"
# os.makedirs(rendered_dir, exist_ok=True)
#
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }
#
# with autocast_context:
#     for out_frame_idx in range(0, len(frame_names)):
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ax.set_title(f"frame {out_frame_idx}")
#         ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#
#         for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#             show_mask(out_mask, ax, obj_id=out_obj_id)
#
#         ax.axis('off')
#         fig.tight_layout()
#         fig.gca().set_position([0, 0, 1, 1])
#
#         # Save the figure as an image using PIL
#         fig.canvas.draw()
#         image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
#         image.save(os.path.join(rendered_dir, f"rendered_frame_{out_frame_idx:05d}.png"))
#
#         plt.close(fig)
