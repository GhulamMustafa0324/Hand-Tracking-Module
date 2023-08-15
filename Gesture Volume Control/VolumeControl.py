import cv2
import time
import HandTrackingModule as htm
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

########Resolution########
wCam, hCam = 720, 480
################

# Initialize VideoCapture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 14, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 14, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # vol = np.interp(length, [45, 200], [minVol, maxVol])
        # Apply a smoothing function (e.g., exponential function) to t
        # smooth_t = t ** 3  # Adjust the exponent value as needed for smoother or faster transitions

        # # Map the smoothed value to the desired volume range
        # smooth_vol = np.interp(smooth_t, [0, 1], [maxVol, minVol])

        # Set the smoothed volume level
        # volume.SetMasterVolumeLevel(vol, None)

        vol = np.interp(length, [45, 125], [minVol, maxVol])
        volBar = np.interp(length, [45, 125], [400, 150])
        volPer = np.interp(length, [45, 125], [0, 100])

        # Apply linear interpolation to map the value to the desired volume range
        # smooth_vol = np.interp(vol, [0, 1], [minVol, maxVol])

        # Set the smoothed volume level
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    cv2.rectangle(img, (35, 150), (85, 400), (255, 0, 0), 2)
    cv2.rectangle(img, (35, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(
        img, f"{int(volPer)}%", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2
    )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, f"FPS {int(fps)}", (10, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    cv2.imshow("Image", img)

    # Wait for a key event and check if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
