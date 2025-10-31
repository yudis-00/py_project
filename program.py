import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "Tidak Dikenali"
        posisi_tangan = "Tidak Terbaca"

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                lm = []
                for landmark in hand_landmarks.landmark:
                    lm.append((int(landmark.x * w), int(landmark.y * h), landmark.z))

                handedness = hand_handedness.classification[0].label  # 'Left' / 'Right'

                tips = [4, 8, 12, 16, 20]
                fingers = []

                # Ambil rata-rata nilai z dari landmark tengah tangan (0, 5, 9, 13, 17)
                palm_z = sum([lm[i][2] for i in [0, 5, 9, 13, 17]]) / 5
                fingertips_z = sum([lm[i][2] for i in tips]) / 5
                diff_z = palm_z - fingertips_z

                # Jika diff_z negatif → telapak menghadap kamera
                # Jika diff_z positif → punggung tangan menghadap kamera
                if diff_z < 0:
                    posisi_tangan = "Depan (Telapak)"
                else:
                    posisi_tangan = "Belakang (Punggung)"

                # ---- Deteksi jari terbuka/tertutup ----
                # Jempol
                if handedness == "Right":
                    fingers.append(1 if lm[tips[0]][0] > lm[tips[0] - 1][0] else 0)
                else:
                    fingers.append(1 if lm[tips[0]][0] < lm[tips[0] - 1][0] else 0)

                # 4 jari lainnya
                for tip in tips[1:]:
                    fingers.append(1 if lm[tip][1] < lm[tip - 2][1] else 0)

                total = fingers.count(1)

                # ---- Tentukan gesture ----
                if total == 0:
                    gesture = "Kepal"
                elif total == 1 and fingers[1] == 1:
                    gesture = "Satu Jari"
                elif total == 2 and fingers[1] == 1 and fingers[2] == 1:
                    gesture = "Dua Jari"
                elif fingers[1:4] == [1, 1, 1] and fingers[0] == 0 and fingers[4] == 0:
                    gesture = "Tiga Jari"
                elif fingers[1:5] == [1, 1, 1, 1] and fingers[0] == 0:
                    gesture = "Empat Jari"
                elif total == 5:
                    gesture = "Lima Jari"
                elif fingers[0] == 1 and sum(fingers[1:]) == 0:
                    gesture = "Jempol"
                elif fingers[0] == 1 and fingers[4] == 1 and sum(fingers[1:4]) == 0:
                    gesture = "Call Me"

        # ---- Tampilkan hasil ----
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # cv2.putText(frame, f"Posisi: {posisi_tangan}", (10, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
