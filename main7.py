import streamlit as st
import cv2
import numpy as np
import random
import time
from PIL import Image, ImageSequence
import mediapipe as mp
from pymongo import MongoClient

# === MongoDB 연결 ===
client = MongoClient(
    "mongodb+srv://jsheek93:j103203j@cluster0.7pdc1.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Cluster0"
)
db = client['game']
users_col = db['game']

# === 게임 설정 ===
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FLY_GIF_PATH = 'fly.gif'
SPEED = 5
GAME_DURATION = 60  # 초

st.title("🦟 모기 잡기 게임")

# === 로그인 / 회원가입 ===
auth_mode = st.sidebar.radio("접속 모드 선택", ['로그인', '회원가입'])
if 'username' not in st.session_state:
    st.session_state.username = None

if auth_mode == '회원가입':
    st.sidebar.subheader("회원가입")
    new_user = st.sidebar.text_input("사용자 이름", key="signup_user")
    new_pass = st.sidebar.text_input("비밀번호", type='password', key="signup_pass")
    if st.sidebar.button("회원가입", key="do_signup"):
        if users_col.find_one({'username': new_user}):
            st.sidebar.error("이미 존재하는 사용자입니다.")
        else:
            users_col.insert_one({
                'username': new_user,
                'password': new_pass,
                'score': None
            })
            st.sidebar.success("회원가입 성공! 로그인 해주세요.")

if auth_mode == '로그인':
    st.sidebar.subheader("로그인")
    user = st.sidebar.text_input("아이디", key="login_user")
    password = st.sidebar.text_input("비밀번호", type='password', key="login_pass")
    if st.sidebar.button("로그인", key="do_login"):
        found = users_col.find_one({'username': user, 'password': password})
        if found:
            st.session_state.username = user
            st.sidebar.success(f"{user}님 환영합니다!")
        else:
            st.sidebar.error("로그인 실패: 잘못된 정보입니다.")

# === 랭킹 보기 ===
if st.sidebar.button("🏆 랭킹 보기", key="show_ranking"):
    st.subheader("🏆 최고 점수 랭킹")
    ranking = users_col.find({'score': {'$ne': None}}).sort('score', -1).limit(10)
    for i, u in enumerate(ranking, start=1):
        st.write(f"{i}등 - {u['username']} : {u['score']}점")

# === 게임 시작 버튼 ===
if st.session_state.username:
    if st.button('게임 시작', key="start_game"):
        # 초기화
        score = 0
        caught = False
        face_penalty = False
        start_time = time.time()
        game_running = True

        # “게임 종료” 버튼을 루프 바깥에서 한 번만 선언
        stop_button = st.button("게임 종료", key="end_game")

        # GIF & Mediapipe 세팅
        pil_gif = Image.open(FLY_GIF_PATH)
        frames = [
            cv2.cvtColor(np.array(f.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
            for f in ImageSequence.Iterator(pil_gif)
        ]
        frame_count = len(frames)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mpDraw = mp.solutions.drawing_utils
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(0)

        # 파리 초기 위치
        fly_x = random.randint(0, WINDOW_WIDTH - frames[0].shape[1])
        fly_y = random.randint(0, WINDOW_HEIGHT - frames[0].shape[0])
        dx = random.choice([-1, 1])
        dy = random.choice([-1, 1])
        frame_index = 0

        # 화면 홀더
        video_pl = st.empty()
        score_pl = st.empty()
        timer_pl = st.empty()

        # 메인 루프
        while game_running and not stop_button:
            elapsed = time.time() - start_time
            if elapsed > GAME_DURATION:
                break  # 시간 종료

            success, img = cap.read()
            if not success:
                st.error('카메라를 열 수 없습니다.')
                break

            img = cv2.flip(img, 1)
            img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 손 인식
            results_hands = hands.process(imgRGB)
            hand_pts = []
            if results_hands.multi_hand_landmarks:
                for handLms in results_hands.multi_hand_landmarks:
                    for idx, lm in enumerate(handLms.landmark):
                        if idx in [4, 8, 12, 16, 20]:
                            h, w, _ = img.shape
                            hand_pts.append((int(lm.x * w), int(lm.y * h)))
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # 파리 움직임
            fly_frame = frames[frame_index % frame_count]
            fh, fw = fly_frame.shape[:2]
            frame_index += 1
            fly_x += dx * SPEED
            fly_y += dy * SPEED
            if fly_x < 0 or fly_x + fw > WINDOW_WIDTH:
                dx *= -1
                fly_x = max(0, min(WINDOW_WIDTH - fw, fly_x))
            if fly_y < 0 or fly_y + fh > WINDOW_HEIGHT:
                dy *= -1
                fly_y = max(0, min(WINDOW_HEIGHT - fh, fly_y))

            # 파리 합성
            overlay = img.copy()
            alpha = fly_frame[:, :, 3] / 255.0
            for c in range(3):
                overlay[fly_y:fly_y+fh, fly_x:fly_x+fw, c] = (
                    alpha * fly_frame[:, :, c] +
                    (1 - alpha) * overlay[fly_y:fly_y+fh, fly_x:fly_x+fw, c]
                )

            # 파리 잡기 판정
            if not caught:
                for (cx, cy) in hand_pts:
                    if fly_x <= cx <= fly_x+fw and fly_y <= cy <= fly_y+fh:
                        score += 1
                        caught = True
                        break
            else:
                if all(not (fly_x <= cx <= fly_x+fw and fly_y <= cy <= fly_y+fh)
                       for (cx, cy) in hand_pts):
                    caught = False

            # 얼굴 감지 감점
            faces = []
            face_res = face_detector.process(imgRGB)
            if face_res.detections:
                h, w, _ = img.shape
                for det in face_res.detections:
                    bb = det.location_data.relative_bounding_box
                    x1, y1 = int(bb.xmin*w), int(bb.ymin*h)
                    x2 = x1 + int(bb.width*w)
                    y2 = y1 + int(bb.height*h)
                    faces.append((x1, y1, x2, y2))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,0,0), 2)

            if not face_penalty:
                for x1,y1,x2,y2 in faces:
                    if (fly_x < x2 and x1 < fly_x+fw and fly_y < y2 and y1 < fly_y+fh):
                        score -= 1
                        face_penalty = True
                        break
            else:
                if all(not (fly_x < x2 and x1 < fly_x+fw and fly_y < y2 and y1 < fly_y+fh)
                       for (x1, y1, x2, y2) in faces):
                    face_penalty = False

            # 화면 그리기
            video_pl.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels='RGB')
            score_pl.text(f'점수: {score}')
            timer_pl.text(f'남은 시간: {int(GAME_DURATION - elapsed)}초')
            time.sleep(0.03)

        cap.release()
        st.success("게임 종료!")

        # 종료 원인에 따라 처리
        if stop_button:
            st.info("게임이 중단되어 점수가 저장되지 않았습니다.")
        else:
            st.write(f"{st.session_state.username}님의 점수는 {score}점입니다.")
            user_data = users_col.find_one({'username': st.session_state.username})
            if user_data['score'] is None or score > user_data['score']:
                users_col.update_one(
                    {'username': st.session_state.username},
                    {'$set': {'score': score}}
                )
                st.success("🎉 최고 기록 갱신!")
            else:
                st.info(f"이전 최고 기록({user_data['score']}점)을 넘지 못했습니다.")
else:
    st.warning("게임을 시작하려면 먼저 로그인하세요!")
