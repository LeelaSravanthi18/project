import cv2
from fer import FER
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import tkinter as tk
from tkinter import messagebox
import webbrowser

detector = FER(mtcnn=True)

camera_running = False
emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
people_count = 0

def start_camera():
    global camera_running
    camera_running = True
    threading.Thread(target=run_camera).start()

def stop_camera():
    global camera_running
    camera_running = False

def run_camera():
    global camera_running, people_count, emotion_counts

    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_emotions(frame)
        people_count = len(results)
        emotion_counts = {key: 0 for key in emotion_counts}

        for result in results:
            bounding_box = result['box']
            emotions = result['emotions']
            x, y, w, h = bounding_box

            max_emotion = max(emotions, key=emotions.get)
            emotion_counts[max_emotion] += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow('Facial Expression Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/start_camera':
            start_camera()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Camera started")
        elif self.path == '/stop_camera':
            stop_camera()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Camera stopped")
        elif self.path == '/people_count':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'people_count': people_count}).encode())
        elif self.path == '/emotion_counts':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(emotion_counts).encode())
        else:
            super().do_GET()

def start_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    httpd.serve_forever()

threading.Thread(target=start_server, daemon=True).start()

root = tk.Tk()
root.title("Facial Expression Analysis")

start_button = tk.Button(root, text="Open Interface", command=lambda: webbrowser.open('http://localhost:8000/interface.html'))
start_button.pack(pady=20)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_camera()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
