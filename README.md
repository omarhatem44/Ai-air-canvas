import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# إعداد MediaPipe لتتبع اليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# إنشاء لوحة رسم فارغة بحجم الكاميرا
ret, frame = cap.read()
h, w, c = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)
white_board = np.ones((h, w, 3), dtype=np.uint8) * 255  # لوحة بيضاء

# قائمة لحفظ نقاط الرسم ومعلومات الألوان
draw_points = deque(maxlen=512)
draw_colors = deque(maxlen=512)  # قائمة لحفظ الألوان لكل نقطة
current_color = (255, 0, 0)  # اللون الافتراضي أزرق
drawing = False  # هل المستخدم يرسم الآن؟

# تعريف الأزرار
buttons = {
    "CLEAR": (10, 10, 100, 50, (0, 0, 0)),
    "PINK": (120, 10, 200, 50, (255, 105, 180)),
    "GREEN": (220, 10, 300, 50, (0, 255, 0)),
    "BLUE": (320, 10, 400, 50, (255, 0, 0)),
    " RED": (420, 10, 500, 50, (0, 0, 255))
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # قلب الصورة أفقيًا
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # رسم الأزرار
    for text, (x1, y1, x2, y2, btn_color) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn_color, -1)
        cv2.putText(frame, text, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # تحديد نقطة رأس السبابة
            index_finger_tip = hand_landmarks.landmark[8]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # التحقق مما إذا كانت اليد فوق زر معين
            for text, (x1, y1, x2, y2, btn_color) in buttons.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    if text == "CLEAR":
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        white_board = np.ones((h, w, 3), dtype=np.uint8) * 255
                        draw_points.clear()
                        draw_colors.clear()
                    else:
                        current_color = btn_color
            
            # التحقق مما إذا كانت اليد في وضع الكتابة
            fingers = [hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in [8, 12, 16, 20]]
            num_fingers = sum(fingers)
            
            if num_fingers == 1:  # إصبع واحد مرفوع → الكتابة
                drawing = True
                draw_points.append((cx, cy))
                draw_colors.append(current_color)
            else:
                if drawing:  # إذا كان المستخدم يرسم ثم رفع يده → إضافة فاصل
                    draw_points.append(None)
                    draw_colors.append(None)
                drawing = False
    
    # رسم الخطوط المتصلة
    for i in range(1, len(draw_points)):
        if draw_points[i - 1] is None or draw_points[i] is None:
            continue
        cv2.line(canvas, draw_points[i - 1], draw_points[i], draw_colors[i], 5)
        cv2.line(white_board, draw_points[i - 1], draw_points[i], draw_colors[i], 5)
    
    # دمج اللوحة مع الإطار الحالي
    output = cv2.addWeighted(frame, 1, canvas, 0.6, 0)
    combined = np.hstack((output, white_board))  # دمج الكاميرا مع اللوحة البيضاء
    
    cv2.imshow("AI Air Canvas", combined)
    
    # حفظ الرسم عند الضغط على "S"
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("drawing.png", canvas)
    
    # الخروج عند الضغط على "Esc"
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
