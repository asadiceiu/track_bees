import os

import cv2
import time

import pandas as pd

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

    nframes = 0
    #cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
    while True:
        #calculate the frame rate
        try:
            frame = cam.get_orb_tracking(draw_kp=draw_kp, draw_detections=draw_detections, draw_tracks=draw_tracks, draw_numbers=draw_numbers)
            nframes += 1
            #cv2.putText(frame, "FPS: {:.2f}".format(cam.fps.fps()), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # resize the frame to fit the screen
            #frame = cv2.resize(frame, (1280, 720))
            # show the frame with option for maximizing the window
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
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

def convert_video(video_path: str, output_path: str):
    """
    Converts the given video to the output path.
    :param video_path:
    :param output_path:
    :return:
    """
    out_size = (640, 360)
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.resize(frame, out_size)
        #cv2.imshow("frame", frame)
        out.write(frame)
        print("{} frames written".format(cap.get(cv2.CAP_PROP_POS_FRAMES)), end="\r")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def yolo_writer():
    import pandas as pd
    csv = pd.read_csv("bboxes.csv", sep='\t', names=["frame", "bee_id", "x", "y", "w", "h"])
    counter = 0
    for i in range(1, 5600):
        rows = csv[csv.frame == i]
        if rows.shape[0] > 0:
            with open("data\\vi_0001_20220725_115507\\obj_train_data\\frame_{:06d}.txt".format(i-1), "w") as f:
                for row in rows.itertuples():
                    f.write(f"0 {row.x:0.6f} {row.y:0.6f} {row.w:0.6f} {row.h:0.6f}\n")
                print(f"{counter} obj_train_data\\frame_{i:06d}.txt")
                counter += 1

def yolo_checker():
    import os
    with open("data\\vi_0001_20220725_115507\\train.txt", "r") as trainfile:
        for line in trainfile:
            # strip line of its newline character and remove the last 3 character
            data = line.strip().split('/')
            fname  = "data/vi_0001_20220725_115507/obj_train_data/"+data[2][:-3]+"txt"
            print(fname)
            # check if the file exists
            if os.path.isfile(fname):
                with open(fname, "r") as txtfile:
                    for row in txtfile:
                        # strip line of its newline character and split it into an array
                        nums = row.strip().split(' ')
                        # check if the array has the correct number of elements
                        if len(nums) != 5:
                            print(f"{fname} has an invalid line: {row}")




if __name__ == "__main__":
    #test_cameras()
    run_detection(video_path='media/vi_0001_20220725_115507.mp4', draw_detections=True, draw_tracks=False, draw_kp=False, draw_numbers=False)
    #convert_video(video_path='media/raspivid90_1.h264', output_path='media/raspivid90_1.h264.avi')
