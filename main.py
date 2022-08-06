import cv2
import time
from camera import VideoCap

def run_detection(video_path: int = 0, draw_detections=True, draw_tracks=False, draw_kp=True, draw_numbers=True):
    """
    Runs the detection on a video.
    :param video_path:
    :param refresh_timeout:
    :return:
    """
    start_time = time.time()
    cam = VideoCap(video_path=video_path, is_direct = True)
    #print camera height and width
    print("Height: ", cam.vs.get(cv2.CAP_PROP_FRAME_HEIGHT), "Width: ", cam.vs.get(cv2.CAP_PROP_FRAME_WIDTH))

    nframes = 0
    while True:
        #calculate the frame rate
        try:
            frame = cam.get_orb_tracking(draw_kp=draw_kp, draw_detections=draw_detections, draw_tracks=draw_tracks, draw_numbers=draw_numbers)
            nframes += 1
            cv2.putText(frame, "FPS: {:.2f}".format(cam.fps.fps()), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            break
    cam.vs.release()
    cv2.destroyAllWindows()
#
def test_cameras():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #print camera height and width
    print("Height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), "Width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# def convert_video(video_path: str, output_path: str):
#     """
#     Converts the given video to the output path.
#     :param video_path:
#     :param output_path:
#     :return:
#     """
#     out_size = (720, 1280)
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, out_size)
#         cv2.imshow("frame", frame)
#         out.write(frame)
#         print("{} frames written".format(cap.get(cv2.CAP_PROP_POS_FRAMES)), end="\r")
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    #test_cameras()
    run_detection()#video_path='media/vi_0001_20220725_115507.mp4', draw_detections=False, draw_tracks=True, draw_kp=True, draw_numbers=True)
    #convert_video(video_path='media/IMG_7931.MOV', output_path='media/IMG_7931.avi')
