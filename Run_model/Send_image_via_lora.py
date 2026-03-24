import serial
import time
import cv2
import numpy as np
import lzma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Uploader")

ARDUINO_PORT = '/dev/ttyUSB0'  # Change to /dev/ttyACM0 or USB1 if needed
BAUDRATE = 115200
CHUNK_SIZE = 56
tracker_id = 2
timeout = 30
is_gray = 1
IMAGE_SIZE = 128

logger.info(f"Connecting to Arduino on {ARDUINO_PORT}...")
ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=2)
time.sleep(2)

logger.info("Waiting for Arduino to say READY...")
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line == "READY":
        logger.info("Arduino is ready!")
        break
    if line != "":
        logger.info(f"Arduino: {line}")

frame = cv2.imread("/home/rapberryiitj/temp/person_det_images/_2025-06-13_17-29-23/tracker_3.jpg")  # Change to your actual image path
if frame is None:
    logger.info("Failed to load image. Check the path.")
    exit()

h, w = frame.shape[:2]
if h < IMAGE_SIZE and w < IMAGE_SIZE:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()
else:
    scale = IMAGE_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    height, width = resized.shape[:2]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()

logger.info(f"Image flattened to {len(flat)} bytes")
flat_c = bytes(flat)
logger.info(f"Image compressed to {len(flat_c)} bytes")

frame_no = 0
total_chunks = (len(flat_c) + CHUNK_SIZE - 1) // CHUNK_SIZE
logger.info(f"Sending image in {total_chunks} chunks of {CHUNK_SIZE}+1 bytes...")

def split_to_3bytes(n):
    return [(n >> (8 * i)) & 0xFF for i in range(3)]

def split_to_2bytes(n):
    return [(n >> (8 * i)) & 0xFF for i in range(2)]

h1 = split_to_2bytes(height)
w1 = split_to_2bytes(width)
s = len(flat_c)
parts = split_to_3bytes(s)
for i in range(15):
    ser.write(bytes([0, 1, 2]))
    st_time = time.time()
    logger.info("sending in dummy bytes")
    
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            logger.info(f"[ARDUINO]: {line}")
            if line == "FREE":
                break

        if time.time() - st_time > 3:
            logger.info("⏳ Timeout: No response from Arduino within 2 seconds.")
            break

    time.sleep(0.1)

l = [tracker_id, w1[0], w1[1], h1[0], h1[1], parts[0], parts[1], parts[2], is_gray]
logger.info(l)

for i in range(4):
    ser.write(bytes(l))
    st_time = time.time()
    logger.info("sending the header")
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            logger.info(f"[ARDUINO]: {line}")
            if line == "FREE":
                break
        if time.time() - st_time > 3:
            logger.info("⏳ Timeout: No response from Arduino within 4 seconds.")
            break
    time.sleep(0.1)

time.sleep(0.1)
prev_time = time.time()

for i in range(0, len(flat_c), CHUNK_SIZE):
    data_chunk = flat_c[i:i + CHUNK_SIZE]
    if len(data_chunk) < CHUNK_SIZE:
        data_chunk = np.pad(np.frombuffer(data_chunk, dtype=np.uint8), (0, CHUNK_SIZE - len(data_chunk)), 'constant')
        data_chunk = data_chunk.tobytes()

    frame_byte = frame_no % 256
    packet = bytes([frame_byte]) + data_chunk

    logger.info(f"Sent frame {frame_no}/{total_chunks - 1} → First byte: {packet[0]} | First 5 bytes: {list(packet[:5])}")

    #for j in range(0, 228, 57):
     #   ser.write(packet[j:57 + j])
    #    st_time = time.time()
  #      while True:
         #   if ser.in_waiting:
         #       line = ser.readline().decode('utf-8').strip()
         #       logger.info(f"[ARDUINO]: {line}")
         #       if line == "FREE":
         #           break
         #   if time.time() - st_time > timeout:
         #       logger.info("⏳ Timeout: No response from Arduino within 4 seconds.")
         #       break
        #time.sleep(0.1)
    ser.write(packet)
    frame_no += 1
    now = time.time()
    time_diff = now - prev_time
    prev_time = now
    logger.info(f"⏱️ Time since last payload: {time_diff:.4f} sec")

    st_time = time.time()
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            logger.info(f"[ARDUINO]: {line}")
            if line == "FREE":
                break
        if time.time() - st_time > timeout:
            logger.info("⏳ Timeout: No response from Arduino within 4 seconds.")
            break
    time.sleep(1)

logger.info("✅ All packets sent.")
