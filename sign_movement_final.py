#!/usr/bin/env python3
import sys
import time
import tty
import termios
import threading
import subprocess
import requests

import cv2
import numpy as np
from flask import Flask, Response
from picarx import Picarx
from vilib import Vilib
from ai_edge_litert.interpreter import Interpreter

# ---------- CONFIG MODELO ----------
MODEL_PATH = "/home/ricardo/modelos/best_sin_flip.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.85
NMS_THRESHOLD = 0.45
STREAM_PORT = 8080

CLASS_NAMES = [
    "avance", "ceder", "contramano", "derecha", "izquierda",
    "pare", "peaton", "peligro", "vel20", "vel30"
]

#CONFIG AUDIO (WiFi hacia la computadora que va a transmitir el audio) 
WINDOWS_IP = "TU_IP"
WINDOWS_PORT = 5001

#CONFIG MOVIMIENTO (calibrado)
DIR_TRIM = -1.9          # correccion de dirección manual para que vaya recto

CRUISE_SPEED = 12
VEL20_SPEED = 5
VEL30_SPEED = 30
SLOW_SPEED = 7
SLOW_TIME = 2.5

STOP_TIME = 2.5

TURN_ANGLE = 30
TURN_TIME = 3.5
TURN_SPEED = 10

CONTRAMANO_TURN_ANGLE = 30
CONTRAMANO_TURN_TIME = 3.5
CONTRAMANO_TURN_SPEED = TURN_SPEED

ACTION_COOLDOWN = 8.0

MIN_BOX_AREA_RATIO_IMMEDIATE = 0.03   

# Umbral especial para ciertas clases 
AREA_OVERRIDES = {
    "contramano": 0.02,"vel30": 0.01, "ceder": 0.04, "pare":0.035, "derecha":0.025
}

IMMEDIATE_CLASSES = {"pare", "peaton", "contramano", "ceder", "peligro", "vel20", "vel30", "avance"}
TURN_CLASSES = {"derecha", "izquierda"}

px = Picarx()
px.set_dir_servo_angle(DIR_TRIM)
px.set_cam_pan_angle(0)
px.set_cam_tilt_angle(0)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modelo cargado.")

annotated_frame = None
frame_lock = threading.Lock()

current_speed = CRUISE_SPEED
last_action_time = {name: 0.0 for name in CLASS_NAMES}

driving = False   # controlado por teclado: 's' arranca, 'p' para

def key_listener():
    """Escucha el teclado sin bloquear: 's' inicia el manejo autonomo, 'p' lo detiene."""
    global driving
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("Presiona 's' para INICIAR el manejo autonomo, 'p' para PARAR.")
        while True:
            ch = sys.stdin.read(1)
            if ch.lower() == 's' and not driving:
                driving = True
                print("[TECLADO] Manejo autonomo INICIADO")
            elif ch.lower() == 'p' and driving:
                driving = False
                px.forward(0)
                px.stop()
                print("[TECLADO] Manejo autonomo DETENIDO")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, orig_w, orig_h):
    predictions = output[0].T
    boxes = predictions[:, :4]
    scores_all = predictions[:, 4:]

    class_ids = np.argmax(scores_all, axis=1)
    confidences = np.max(scores_all, axis=1)

    mask = confidences >= CONF_THRESHOLD
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            results.append({
                "box": (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                "conf": float(confidences[i]),
                "class_id": int(class_ids[i]),
                "area_ratio": float(w[i] * h[i]),
            })
    return results

def anunciar(texto):
    wav_path = "/tmp/tts.wav"
    try:
        subprocess.run(["espeak-ng", "-v", "es", "-w", wav_path, texto], check=True)
        with open(wav_path, "rb") as f:
            requests.post(f"http://{WINDOWS_IP}:{WINDOWS_PORT}/play", data=f.read(), timeout=3)
    except Exception as e:
        print(f"No se pudo anunciar: {e}")

#VIDEO STREAM
app = Flask(__name__)

def gen_frames():
    while True:
        with frame_lock:
            frame = None if annotated_frame is None else annotated_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True, debug=False, use_reloader=False)

# MOVIMIENTO
def action_stop_and_wait():
    px.stop()
    time.sleep(STOP_TIME)

def action_advance_then_stop(avance_time=0.5):
    """Sigue avanzando un rato mas antes de frenar (para acercarse mas al cartel)."""
    px.set_dir_servo_angle(DIR_TRIM)
    px.forward(current_speed)
    time.sleep(avance_time)
    action_stop_and_wait()

def action_slow_down():
    px.set_dir_servo_angle(DIR_TRIM)
    px.forward(SLOW_SPEED)
    time.sleep(SLOW_TIME)

def action_set_speed(nueva_velocidad):
    global current_speed
    current_speed = nueva_velocidad

def action_turn(direccion, avance_time=0.5):
    px.set_dir_servo_angle(DIR_TRIM)
    px.forward(current_speed)
    time.sleep(avance_time)
    angulo = TURN_ANGLE if direccion == "derecha" else -TURN_ANGLE
    px.set_dir_servo_angle(angulo)
    px.forward(TURN_SPEED)
    time.sleep(TURN_TIME)
    px.set_dir_servo_angle(DIR_TRIM)

def action_turn_contramano():
    px.set_dir_servo_angle(CONTRAMANO_TURN_ANGLE)
    px.forward(CONTRAMANO_TURN_SPEED)
    time.sleep(CONTRAMANO_TURN_TIME)
    px.set_dir_servo_angle(DIR_TRIM)

IMMEDIATE_ACTIONS = {
    "pare": action_advance_then_stop,
    "peaton": action_advance_then_stop,
    "contramano": action_turn_contramano,
    "ceder": action_advance_then_stop,
    "peligro": action_advance_then_stop,
    "vel20": lambda: action_set_speed(VEL20_SPEED),
    "vel30": lambda: action_set_speed(VEL30_SPEED),
    "avance": lambda: action_set_speed(CRUISE_SPEED),
}

def procesar_detecciones(detections):
    now = time.time()

    if len(detections) > 1:
        detections = [min(detections, key=lambda d: d["area_ratio"])]

    for det in detections:
        clase = CLASS_NAMES[det["class_id"]]
        print(f"[DETECCION] {clase} (conf={det['conf']:.2f}, area={det['area_ratio']:.3f})")

        if clase in IMMEDIATE_CLASSES:
            umbral = AREA_OVERRIDES.get(clase, MIN_BOX_AREA_RATIO_IMMEDIATE)
            if det["area_ratio"] >= umbral:
                if now - last_action_time[clase] >= ACTION_COOLDOWN:
                    print(f"[ACCION] {clase} (conf={det['conf']:.2f}, area={det['area_ratio']:.2f})")
                    anunciar(f"Señal detectada: {clase}")
                    IMMEDIATE_ACTIONS[clase]()
                    last_action_time[clase] = now

        elif clase in TURN_CLASSES:
            umbral = AREA_OVERRIDES.get(clase, MIN_BOX_AREA_RATIO_IMMEDIATE)
            if det["area_ratio"] >= umbral:
                if now - last_action_time[clase] >= ACTION_COOLDOWN:
                    print(f"[ACCION] Girando a la {clase} (area={det['area_ratio']:.2f})")
                    anunciar(f"Señal detectada: {clase}")
                    action_turn(clase)
                    last_action_time[clase] = now

def inference_loop():
    global annotated_frame

    Vilib.camera_start(vflip=False, hflip=False)
    time.sleep(1)

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()
    print(f"Stream en: http://<IP_DEL_ROBOT>:{STREAM_PORT}/video")

    try:
        while True:
            frame = Vilib.img
            if frame is None:
                time.sleep(0.05)
                continue

            orig_h, orig_w = frame.shape[:2]
            input_tensor = preprocess(frame)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            detections = postprocess(output, orig_w, orig_h)

            drawn = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                clase = CLASS_NAMES[det["class_id"]]
                label = f'{clase} {det["conf"]:.2f} area={det["area_ratio"]:.3f}'
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(drawn, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # velocidad actual, siempre visible en el stream 
            cv2.putText(drawn, f"Velocidad: {current_speed}", (10, orig_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if not driving:
                cv2.putText(drawn, "Presiona 's' para iniciar", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                with frame_lock:
                    annotated_frame = drawn
                continue

            with frame_lock:
                annotated_frame = drawn

            procesar_detecciones(detections)

            px.set_dir_servo_angle(DIR_TRIM)
            px.forward(current_speed)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            Vilib.camera_close()
        except Exception:
            pass
        px.stop()
        print("Manejo autónomo detenido.")

if __name__ == "__main__":
    inference_loop()
