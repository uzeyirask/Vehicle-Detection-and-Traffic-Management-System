import pyodbc
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

#Initialize video capture and model
cap = cv2.VideoCapture("../Videos/trafikOn4.mp4")   #Video kaynagi
model = YOLO('../Yolo-Weights/yolov8n.pt')  #Kullanilan YOLO modelinin secimi

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4,720)
# model = YOLO('../Yolo-Weights/yolov8n.pt')

# Sinif adlarini tanimlama ve Mask kullanimi icin
classNames = ["car", "motorbike", "bus", "truck"]
mask = cv2.imread("mask.png")

# İzleyiciyi ve sınırları başlat
trackers = Sort(max_age =20, min_hits=3, iou_threshold=0.3)
limits = [100, 397, 600, 397]
totalCount = []

# Veritabanı bağlantı bilgileri
server = 'MUA-PC\SQLEXPRESS'
database = 'ArabaSayaciDB'
username = ''  # Veritabanı kullanıcı adı
password = ''  # Veritabanı şifresi (-)

# Veritabanı bağlantısını oluştur
conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

# Veritabanı bağlantısını kullanarak bir cursor (imleç) oluştur
cursor = conn.cursor()

while True:
    success, img = cap.read()
    if not success:
        break

    imgCar = cv2.imread("img_1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgCar, (0, 0))
    imgRegion = cv2.bitwise_and(img, mask)
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Convert Tensor to float
            cls = int(box.cls[0])

            if cls < len(classNames) and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9)

    resultsTracker = trackers.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    vehicleCount = 0
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and id not in totalCount:
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            totalCount.append(id)

        vehicleCount += 1

    cv2.putText(img, str(len(totalCount)), (250, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    if vehicleCount >= 4:
        cv2.putText(img, "Yogun Trafik!", (400, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 8)

    cv2.imshow("Image", img)
    print(totalCount)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break  # 'q' tuşuna basıldığında döngüyü sonlandır

# Veritabanına son totalCount değerini gönder
if totalCount:
    last_id = totalCount[-1]
    cursor.execute("INSERT INTO ArabaSayisi (KameraID, GecenArabaSayisi, KayitTarihi) VALUES (?, ?, GETDATE())", (1, last_id))
    conn.commit()  # Veritabanı işlemi tamamlandı

# Bağlantıyı kapat
conn.close()
cap.release()
cv2.destroyAllWindows()