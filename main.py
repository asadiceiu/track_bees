import cv2

from camera import VideoCap

def run_detection(video_path: str, refresh_timeout: int = 500):
    """
    Runs the detection on a video.
    :param video_path:
    :param refresh_timeout:
    :return:
    """
    cam = VideoCap(video_path=video_path, refresh_timeout=refresh_timeout, is_direct = True)
    while True:
        frame = cam.get_orb_tracking()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    run_detection(video_path='media/vi_0000_20220803_171422.mp4', refresh_timeout=3000)
