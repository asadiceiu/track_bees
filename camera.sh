# raspberry pi camera script
for i in {1..10}
do
  raspistill -o /home/pi/Desktop/image$i.jpg
done

for i in {1..10}
# file name is the current date and time and set it to a variable
do
  FILENAME=$(date +%Y%m%d%H%M%S)
  # print the file name to the screen
  echo $FILENAME
  sudo raspivid -o /home/pi/raspivid$FILEMAME.h264 -t 300000 -w 1280 -h 720 -fps 60 -b 1200000 # 300000 = 5 minutes
  sudo MP4Box -add /home/pi/raspivid$FILENAME.h264 /home/pi/raspivid$FILENAME.mp4
  sudo rm /home/pi/raspivid$FILEMAME.h264
done
