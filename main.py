import os
import cv2
import gdown
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single,fine_face_swap

# Initialize Flask app
app = Flask(__name__)

# Initialize the FaceAnalysis app
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown if not exists
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Create a temporary directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')
    
if not os.path.exists('videos'):
    os.makedirs('videos')

@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    if 'video' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'No video or image file provided'}), 400

    video = request.files['video']
    image = request.files['image']
    
    video_filename = secure_filename(video.filename)
    image_filename = secure_filename(image.filename)
    
    video_path = os.path.join('uploads', video_filename)
    image_path = os.path.join('uploads', image_filename)
    
    # Save the uploaded files
    video.save(video_path)
    image.save(image_path)
    
    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
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
    frame_skip = 1  # Skip every frame_skip frames to speed up the process
    for i in range(0, len(frame_list), frame_skip):
        frame = cv2.imread(frame_list[i])
        output_frame = fine_face_swap(frame_list[i], image_path, face_app, swapper, enhance=False, enhancer='REAL-ESRGAN 2x', device="cpu")
        cv2.imwrite(frame_list[i], output_frame)
    
    # Reassemble the frames into a video
    output_video_path = f'videos/output_{video_filename}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    
    # Clean up temporary frames
    for frame_path in frame_list:
        os.remove(frame_path)
    os.rmdir('frames')
    
    return jsonify({'output_video_path': output_video_path}), 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
