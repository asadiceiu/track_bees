# raspberry pi camera script
for i in `seq 1 10`
# file name is the current date and time and set it to a variable
do
  FILENAME=$(date +%Y%m%d%H%M%S)
  # print the file name to the screen
  echo $FILENAME
  sudo raspivid -o /home/pi/raspivid$FILEMAME.h264 -t 300000 -w 1280 -h 720 -fps 60 -b 1200000 # 300000 = 5 minute
done
