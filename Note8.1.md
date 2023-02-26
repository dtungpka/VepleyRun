



Thu cả RGB và skeleton (raw) + noted

Áp dụng nhiều mạng DL (so sánh) - CNN

Trên skeleton: 

- CNN
- Rule of base : Kiểm tra đk xem phân cluster đc k 
- coi như là 1 vector => SVM, etc,.
- Fusion ML + DL: Early fusion, Late fusion, (Middle fusion)

Ưu tiên thu dữ liệu (full)


# Chuẩn hóa theo joint 0 (Cổ tay)
=> Cho vào thành 1 vector đặc trưng


# kết hợp cả tọa độ và góc khớp tay
## Scale lại tọa độ

- early fusion:
Kết hợp trước khi đưa vào mạng
Lưu ý về chuẩn hóa
- late fusion:
Huấn luyện 2 mạng riêng biệt,
 sau đó kết hợp lại
- middle fusion


- Riêng trên ảnh RGB:
### Huấn luyện riêng 1 cách khác: YOLO.


-- Huấn luyện nhận dạng luôn action từ ảnh


- Kết hợp ảnh RGB với skeleton: 
early fusion, late fusion, middle fusion
Nên sử dụng late fusion.

- Thử kết hợp các phương pháp trên theo cách khác nhau

Lưu lại thời gian, các tham số. 

các kết quả để so sánh


## Việc hiện tại:
thu dữ liệu
huấn luyện model bằng những phương pháp trên
tích hợp model tốt nhất vào game
Lưu lại số liệu, tham số.

Lịch họp sáng thứ 7 hàng tuần: 9h


### 12/2
Đọc các bài báo về các model, xem cải tiến , kết hợp

Thử thay đổi góc của joint: sử dụng chung 1 tọa độ joint 0

Tìm các bài báo: đề xuất đặc trưng của tình trạng bàn tay (hand posture), skeleton, feature extraction, hand posture classifition/recognition





# 18/2



Trong tuần này:

- Thu dữ liệu, gộp lại
- Train theo 2 phương pháp



# 26/2
## VD: thu 10 ng: chia 8 nguoi train, 2 nguoi test
Moi ng thu ~10 lan, train/test =  8/2

- Muc tieu: nhan dang posture tung frame



Xu li sau:
- Tim cach xu li nhieu
- Tim cach xu li nhieu nguoi













