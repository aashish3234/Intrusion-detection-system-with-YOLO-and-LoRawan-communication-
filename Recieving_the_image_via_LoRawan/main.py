import os
import time
import json
import base64
import ssl
import numpy as np
import paho.mqtt.client as mqtt
from collections import defaultdict
from queue import Queue
import threading
import logging
from datetime import datetime

# ==============================
# LOGGER SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURATION
# ==============================
APP_ID = "iitjammu-lora-devices"
DEVICE_ID = "uct-lora"
API_KEY = "NNSXS.NKBGK3QR4HKZ6W6CUT7LCNU2WNEYWHA5LWVOVSQ.KBTONZXJ5C375TASPVAEW57RHESX5UISHYPUBRMMWIUIPUXR6CGA"
MQTT_HOST = "au1.cloud.thethings.network"
MQTT_PORT = 8883
TOPIC = f"v3/{APP_ID}@ttn/devices/{DEVICE_ID}/up"

SAVE_FOLDER = "/home/aashish/lorawan"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==============================
# GLOBALS
# ==============================
image_data = defaultdict(lambda: None)
last_received_time = time.time()
IMAGE_SIZE = 128 * 128
FRAME_SIZE = 56
FIRST_IMAGE_FRAME = 0
TOTAL_FRAMES = (IMAGE_SIZE // FRAME_SIZE) + (1 if IMAGE_SIZE % FRAME_SIZE else 0)
image_counter = 0
height = 0
width = 0
size = 0
y = 0
frames_needed = 0
is_gray=1
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# ==============================
# QUEUE SETUP
# ==============================
message_queue = Queue()

def reconstruct_from_3bytes(parts):
    return sum([parts[i] << (8 * i) for i in range(3)])

def reconstruct_from_2bytes(parts):
    return sum([parts[i] << (8 * i) for i in range(2)])

def process_message(msg):
    global last_received_time, image_data, image_counter, height, width, size, y, FRAME_SIZE, frames_needed,is_gray,now_str
    last_received_time = time.time()

    try:
        payload = json.loads(msg.payload.decode())
        uplink = payload.get("uplink_message", {})
        frm_payload = uplink.get("frm_payload", "")
        if not frm_payload:
            return

        raw_bytes = base64.b64decode(frm_payload)
        if len(raw_bytes) < 9:
            logger.info("found dummy")
            return

        if len(raw_bytes) == 9:
            image_counter = 1
            tracker_id = raw_bytes[0]
            w = [raw_bytes[1], raw_bytes[2]]
            h = [raw_bytes[3], raw_bytes[4]]
            x = [raw_bytes[5], raw_bytes[6], raw_bytes[7]]
            is_gray = raw_bytes[8]
            width = reconstruct_from_2bytes(w)
            height = reconstruct_from_2bytes(h)
            size = reconstruct_from_3bytes(x)
            frames_needed = (size + FRAME_SIZE - 1) // FRAME_SIZE
            logger.info("recived header")
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            return

        frame_no = raw_bytes[0]
        data = raw_bytes[1:]

        if len(data) != FRAME_SIZE:
            logger.warning(f"Invalid frame size for frame {frame_no}, got {len(data)} bytes")
            return

        if frame_no < 254 and y:
            frame_no += 256
        if image_data[frame_no] is None:
            image_data[frame_no] = list(data)
            logger.info(f"Frame {frame_no} received.")
            if frames_needed and size and width:
                logger.info("")
            else:
                logger.info("wrong")
            


            txt_log_file = os.path.join(SAVE_FOLDER, f"image_{now_str}_frames.txt")
            with open(txt_log_file, "a") as f:
                data_str = ", ".join(str(b) for b in data)
                f.write(f"Frame {frame_no:03}: {data_str}\n")

        if (frames_needed <= frame_no + 1) and image_counter:
            reconstruct_image_from_txt(is_gray, txt_log_file, width, height, size)
            image_counter = 0

        if frame_no > 254:
            y += 1
    except Exception as e:
        logger.error("Error handling message:", exc_info=True)

def process_thread():
    while True:
        msg = message_queue.get()
        if msg is None:
            break
        process_message(msg)

worker = threading.Thread(target=process_thread, daemon=True)
worker.start()

# ==============================
# MQTT CALLBACKS
# ==============================
def on_connect(client, userdata, flags, rc, properties=None):
    logger.info(f"Connected with result code: {rc}")
    client.subscribe(TOPIC, qos=1)
    logger.info(f"Subscribed to: {TOPIC}")

def on_message(client, userdata, msg):
    message_queue.put(msg)

# ==============================
# IMAGE PROCESSING FUNCTION
# ==============================
def reconstruct_image_from_txt(is_gray, txt_path, width, height, compressed_size):
    global now_str
    import cv2
    import lzma
    import re

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    frames = []
    received_frames = set()
    for line_no, line in enumerate(lines):
        match = re.match(r"Frame (\d+): (.*)", line)
        if match:
            frame_no = int(match.group(1))
            byte_str = match.group(2)
            byte_list = [int(x.strip()) for x in byte_str.split(',') if x.strip()]
            frames.append((frame_no, byte_list))
            received_frames.add(frame_no)

    frames.sort(key=lambda x: x[0])

    reconstructed_bytes = []
    expected_frames = (compressed_size + 226) // 227

    for frame_no in range(expected_frames):
        if frame_no in received_frames:
            byte_list = next(bl for fn, bl in frames if fn == frame_no)
        else:
            logger.warning(f"Missing frame: {frame_no}, filling with 255s")
            byte_list = [255] * 227
        reconstructed_bytes.extend(byte_list)
        if len(reconstructed_bytes) >= compressed_size:
            break

    compressed_data = np.array(reconstructed_bytes[:compressed_size], dtype=np.uint8)
    logger.info(f"Reconstructed compressed data length: {len(compressed_data)} bytes (expected: {compressed_size})")

    total_pixels = width * height
    flat_arr = compressed_data

    if is_gray == 1:
        if len(flat_arr) != total_pixels:
            logger.error(f"Size mismatch after decompression: expected {total_pixels}, got {len(flat_arr)}")
            return
        reconstructed_image = flat_arr.reshape((height, width))
    else:
        expected_len = total_pixels * 3
        if len(flat_arr) != expected_len:
            logger.error(f"Size mismatch after decompression: expected {expected_len}, got {len(flat_arr)}")
            return
        r = flat_arr[0:total_pixels].reshape((height, width))
        g = flat_arr[total_pixels:2 * total_pixels].reshape((height, width))
        b = flat_arr[2 * total_pixels:3 * total_pixels].reshape((height, width))
        reconstructed_image = cv2.merge((b, g, r))
    # now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(txt_path), f"reconstructed_image_{now_str}.jpg")
    cv2.imwrite(output_path, reconstructed_image)
    logger.info(f"Reconstructed image saved at: {output_path}")

# ==============================
# MAIN SETUP
# ==============================
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(APP_ID, API_KEY)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT)
client.loop_start()

logger.info("Listening for incoming TTN messages...")

while True:
    time.sleep(5)
