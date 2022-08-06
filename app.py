import sys

import cv2
from flask import Flask, render_template, Response
from camera import VideoCap

app = Flask(__name__)
cam = None  # VideoCap(0, refresh_timeout=500) #video_path='media/vi_0000_20220725_122016.mp4', refresh_timeout=500)


def camera_frame(camera, video_type: str = "index"):
    """
    Returns a camera frame.
    :param camera:
    :param video_type:
    :return:
    """
    camera = VideoCap(0, refresh_timeout=500) if camera is None else camera
    while True:
        if video_type == "index":
            frame = camera.get_frame()
        elif video_type == "track":
            frame = camera.get_orb_tracking()
        elif video_type == "backgroundsubtraction":
            frame = camera.get_background()
        else:
            frame = None
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/')
def index():
    title = "Video"
    return render_template('index.html', title=title)


@app.route('/backgroundsubtraction')
def backgroundsubtraction():
    title = "Background Subtraction"
    return render_template('backgroundsubtraction.html', title=title)


@app.route('/track')
def track():
    title = "Tracking Bees"
    return render_template('track.html', title=title)


@app.route('/video/<string:video_type>')
def video(video_type: str = "index"):
    return Response(camera_frame(cam, video_type), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # get camera refresh rate as and argument
    refresh_timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    cam = VideoCap(0, refresh_timeout=refresh_timeout, is_direct=False, height=height, width=width)
    app.run(debug=False, host='0.0.0.0')
