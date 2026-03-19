# BTL--trituenhantao
# Hướng dẫn cài đặt và chạy chương trình

🔹 Bước 1: Cài đặt Anaconda

Truy cập: https://www.anaconda.com

Tải và cài đặt

Mở Anaconda Prompt

🔹 Bước 2: Tạo môi trường Python 3.10
conda create -n mask_env python=3.10

→ Nhấn y để xác nhận

🔹 Bước 3: Kích hoạt môi trường
conda activate mask_env

→ Nếu thấy (mask_env) là OK

🔹 Bước 4: Cài thư viện cần thiết
<pre> pip install tensorflow
pip install opencv-python
pip install numpy
pip install imutils </pre>

🔹 Bước 5: Mở thư mục project

Theo ảnh của bạn:

cd D:\k225480106094_Nông_Hồ_Nhật_BTL-TTNT_Phatienkhutrang

Cấu trúc project:

face_detector/
images/
detect_mask_image.py
mask_detector.h5

🚀 HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
🔹 Bước 1: Kiểm tra model và file

Đảm bảo có:

mask_detector.h5 → model khẩu trang

face_detector/ → model phát hiện khuôn mặt

images/ → chứa ảnh test

🔹 Bước 2: Chạy chương trình
python detect_mask_image.py

🔹 Bước 3: Kết quả

Chương trình sẽ:

Load model mask_detector.h5

Dùng SSD để detect khuôn mặt

Cắt vùng mặt → đưa vào model

Dự đoán:

Mask

No Mask

Hiển thị ảnh với bounding box + label
