import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# Load model khẩu trang
# ==========================
model = load_model("mask_detector.h5", compile=False)

# ==========================
# Load model phát hiện khuôn mặt
# ==========================
prototxt = "face_detector/deploy.prototxt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxt, weights)

# ==========================
# Đọc ảnh
# ==========================
image = cv2.imread("images/test4.jpg")

if image is None:
    print("❌ Không đọc được ảnh")
    exit()

# ✅ Resize trước để chữ không thay đổi
image = cv2.resize(image, (1000, 750))

(h, w) = image.shape[:2]

# ==========================
# Chuẩn hóa ảnh
# ==========================
blob = cv2.dnn.blobFromImage(
    image,
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

faceNet.setInput(blob)
detections = faceNet.forward()

# ==========================
# Cấu hình chữ
# ==========================
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

# ==========================
# Duyệt từng khuôn mặt
# ==========================
for i in range(0, detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

        face = image[startY:endY, startX:endX]

        if face.size == 0:
            continue

        # ==========================
        # Chuẩn hóa cho model
        # ==========================
        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        mask, withoutMask = preds

        if mask > withoutMask:
            label = "Deo khau trang"
            color = (0, 255, 0)
            score = mask
        else:
            label = "Khong deo khau trang"
            color = (0, 0, 255)
            score = withoutMask

        text = f"{label}: {score*100:.2f}%"

        # ==========================
        # Vẽ khung khuôn mặt
        # ==========================
        cv2.rectangle(
            image,
            (startX, startY),
            (endX, endY),
            color,
            2
        )

        # ==========================
        # Tính kích thước chữ
        # ==========================
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        textX = startX
        textY = startY - 10

        # nếu vượt mép phải
        if textX + text_w > w:
            textX = w - text_w - 10

        # nếu vượt mép trên
        if textY - text_h < 0:
            textY = startY + text_h + 10

        # ==========================
        # Vẽ nền chữ
        # ==========================
        cv2.rectangle(
            image,
            (textX, textY - text_h - 5),
            (textX + text_w + 5, textY + 5),
            (0, 0, 0),
            -1
        )

        # ==========================
        # Vẽ chữ
        # ==========================
        cv2.putText(
            image,
            text,
            (textX + 2, textY),
            font,
            font_scale,
            color,
            thickness
        )

# ==========================
# Hiển thị ảnh
# ==========================
cv2.namedWindow("Mask Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Mask Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()