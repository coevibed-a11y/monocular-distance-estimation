# infer_stream.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
from flask import Flask, Response, render_template_string

import config as cfg
from detector import VehicleDetector
from visualizer import TrafficVisualizer

app = Flask(__name__)

print(">>> 시스템 초기화 중...(Flask MJPEG 모드)")
detector = VehicleDetector()
visualizer = TrafficVisualizer()

# 🔹 전역 VideoCapture (한 번만 열어서 계속 사용)
cap = cv2.VideoCapture(cfg.VIDEO_PATH)
if not cap.isOpened():
    print("Error: 영상을 열 수 없습니다.")
    raise RuntimeError("영상 로드 실패")

# 🔹 일시정지 / 재생 상태 플래그
is_paused = False

# 🔹 마지막으로 그린 JPEG (일시정지 때 재사용)
latest_jpeg = None


def gen_frames():
    """
    /video_feed: 브라우저 하나의 MJPEG 스트림
    - is_paused == True면 새 프레임을 안 읽고 latest_jpeg만 계속 보냄
    """
    global latest_jpeg, is_paused, cap

    prev_time = 0.0
    fps = 0.0
    last_vis_time = 0.0
    vis_interval = 1.0 / cfg.VISUALIZATION_FPS

    while True:
        curr_time = time.time()

        # ------------------------------
        # 1) 프레임 읽기 (재생 중일 때만)
        # ------------------------------
        if not is_paused:
            ret, frame = cap.read()

            # 영상 끝나면 처음부터 (원하면 여기서 break로 완전 종료해도 됨)
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            input_frame = cv2.resize(frame, cfg.TARGET_SIZE)

            # ---- 추론 ----
            results, _ = detector.run(input_frame)

            # ---- FPS 계산 (추론 성능) ----
            term = curr_time - prev_time
            if term > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / term)
            prev_time = curr_time

            # ---- 시각화 FPS 제한 ----
            if (curr_time - last_vis_time) >= vis_interval:
                last_vis_time = curr_time

                display_frame = visualizer.draw_results(input_frame.copy(), results)

                fps_color = (0, 255, 0) if fps > 15 else (0, 0, 255)
                cv2.putText(
                    display_frame,
                    f"System FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    fps_color,
                    2
                )

                info = f"Scale: {cfg.GEO_SCALE} | Vis Limit: {cfg.VISUALIZATION_FPS}fps"
                cv2.putText(
                    display_frame,
                    info,
                    (20, cfg.TARGET_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                ret_jpg, buffer = cv2.imencode(".jpg", display_frame)
                if ret_jpg:
                    latest_jpeg = buffer.tobytes()
        else:
            # 일시정지 상태면 새 프레임 안 읽음
            # latest_jpeg 를 그대로 사용
            time.sleep(0.01)

        # ------------------------------
        # 2) latest_jpeg를 MJPEG로 전송
        # ------------------------------
        if latest_jpeg is None:
            # 아직 프레임 준비 안 됐으면 패스
            continue

        try:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                latest_jpeg +
                b"\r\n"
            )
        except GeneratorExit:
            # 클라이언트가 탭 닫으면 여기로 빠짐
            print(">>> 클라이언트 스트림 종료")
            break
        except Exception as e:
            print(f">>> 스트림 전송 중 예외: {e}")
            break


# ============================
#  라우트
# ============================

HTML_PAGE = """
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>Traffic Analysis MJPEG</title>
    <style>
      body {
        margin: 0;
        background: #111;
        color: #eee;
        font-family: Arial, sans-serif;
      }
      .wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px;
      }
      h1 {
        font-size: 24px;
        margin: 10px 0 6px 0;
      }
      .controls {
        margin: 8px 0 12px 0;
        display: flex;
        gap: 8px;
      }
      button {
        padding: 6px 14px;
        font-size: 14px;
        border-radius: 4px;
        border: 1px solid #444;
        background: #222;
        color: #eee;
        cursor: pointer;
      }
      button:hover {
        background: #333;
      }
      button:active {
        background: #555;
      }
      img {
        max-width: 100%;
        height: auto;
        border: 2px solid #444;
        background: #000;
      }
      .info {
        margin-top: 8px;
        font-size: 13px;
        color: #aaa;
        text-align: center;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Traffic Analysis (Flask MJPEG)</h1>

      <div class="controls">
        <button onclick="resumeStream()">▶ 재생</button>
        <button onclick="pauseStream()">⏸ 일시정지</button>
      </div>

      <!-- src는 계속 /video_feed 로 유지 -->
      <img id="stream" src="/video_feed" alt="Traffic Stream">

      <div class="info">
        ▶ / ⏸ 는 서버 상태만 바꾸고,<br>
        화면은 현재 프레임에서 멈췄다가 그 이후부터 이어서 재생됩니다.
      </div>
    </div>

    <script>
      function pauseStream() {
        fetch('/control/pause', { method: 'POST' });
      }

      function resumeStream() {
        fetch('/control/resume', { method: 'POST' });
      }
    </script>
  </body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    """MJPEG 스트림"""
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/control/pause", methods=["POST"])
def control_pause():
    global is_paused
    is_paused = True
    print(">>> 일시정지 요청")
    return ("OK", 200)


@app.route("/control/resume", methods=["POST"])
def control_resume():
    global is_paused
    is_paused = False
    print(">>> 재생 요청")
    return ("OK", 200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
