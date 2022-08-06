from webcamstream import WebcamVideoStream, FPS

import cv2

vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    _, frame = vs.read()
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    fps.update()
    key = cv2.waitKey(10) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.release()
