import os
import matplotlib.pyplot as plt
import gdown
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single,fine_face_swap
import cv2
# Initialize the FaceAnalysis app
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Paths to input video and images
input_video_path = 'videos/gym_guy_reel_22.mp4'
output_video_path = 'videos/gym_guy_reel_22.mp4'
img2_fn = 'images/srk.jpg'  # Face to swap in

# Create a temporary directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Extract frames from the video
cap = cv2.VideoCapture(input_video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_list = []
for i in range(frame_count):
    ret, frame = cap.read()
    if ret:
        frame_path = f'frames/frame_{i:04d}.jpg'
        cv2.imwrite(frame_path, frame)
        frame_list.append(frame_path)
    else:
        break
cap.release()

# Perform face swapping on each frame
frame_skip =    1  # Skip every 8 frames to speed up the process
for i in range(0, len(frame_list), frame_skip):
    frame = cv2.imread(frame_list[i])
    output_frame = fine_face_swap(frame_list[i], img2_fn, app, swapper, enhance=False, enhancer='REAL-ESRGAN 2x', device="cpu")
    cv2.imwrite(frame_list[i], output_frame)

# Reassemble the frames into a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for frame_path in frame_list:
    frame = cv2.imread(frame_path)
    out.write(frame)
out.release()

# # Optional: Add audio from the original video to the output video
# original_clip = mp.VideoFileClip(input_video_path)
# output_clip = mp.VideoFileClip(output_video_path)
# output_clip_with_audio = output_clip.set_audio(original_clip.audio)
# output_clip_with_audio.write_videofile(output_video_path, codec='libx264')

# Clean up temporary frames
for frame_path in frame_list:
    os.remove(frame_path)
os.rmdir('frames')
# fine_face_swap(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 8x',device="cpu")