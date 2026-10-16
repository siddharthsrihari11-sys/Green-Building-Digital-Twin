#   pip install opencv-python requests
#   python send_cam.py http://HOST_IP:5000 1        (use https:// if the host started with HTTPS=1)
import sys, time, cv2, requests, urllib3
urllib3.disable_warnings()                      # the host's self-signed certificate is expected
server, cam = sys.argv[1].rstrip("/"), sys.argv[2]
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit("Cannot open the webcam - close other apps that use it.")
print(f"Sending webcam to {server} as camera {cam}. Press Ctrl+C to stop.")
sent = 0
while True:
    ok, frame = cap.read()
    if ok:
        ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 60])
        try:
            requests.post(f"{server}/api/remote_cam/{cam}", data=buf.tobytes(), timeout=3, verify=False)
            sent += 1
            if sent % 40 == 0: print(f"{sent} frames sent")
        except requests.RequestException as e:
            print("Cannot reach the server:", e); time.sleep(1)
    time.sleep(0.08)