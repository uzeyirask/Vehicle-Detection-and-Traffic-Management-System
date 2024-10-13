import cap
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/trafikOn4.mp4")
# cap = cv2.VideoCapture("../Videos/trafikArka2.mp4") #Diger yol icin


model = YOLO('../Yolo-Weights/yolov8n.pt')  # Hangi yolo modelini kullacagimi buradan sectim

# Sadece ilgilendiğim sınıflar
classNames = ["car", "motorbike", "bus", "truck"]

mask = cv2.imread("mask.png")  # mask icin yapilan 1. islem

# Tacking
trackers = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# limits = [200, 397, 600, 397]
limits = [100, 397, 600, 397]  # Diger yol icin

totalCount = []

while True:  # Cerceveler icin gerekli islemleri yapiyoruz cerceve sekli cerceve rengi gibi

    success, img = cap.read()
    if not success:
        break

    imgCar = cv2.imread("img_1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgCar, (0, 0))

    imgRegion = cv2.bitwise_and(img, mask)  # mask icin yapilan 2. hamle
    result = model(imgRegion, stream=True)  # imgRegion kullanarak masktaki fotoyu kullanarak maskeliyoruz

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Kutulari olusturma
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)
            # Kutu sinirlari ve fotograftaki nesnelerin dogruluk degerleri
            conf = math.ceil((box.conf[0] * 100)) / 100  # Ekranda gosterilen nesnelerin
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1) #Conf ile aldigim degeri(dogruluk degerini ekranda open cv araciligiyla gosterir) ve ekranda gosterilen sayisal degerin sinirlarda kalmasini sagladim
            # ClassName
            cls = int(box.cls[0])

            # Eğer cls değeri classNames listesinin sınırları içindeyse işlemi gerçekleştir
            if cls < len(classNames):
                currentClass = classNames[cls]
                # Bu blok sayesinde sadece car, bus, motorbike, truckları alır ve bunların doğruluk değeri 0.3'den fazla olmalıdır
                if currentClass in classNames and conf > 0.3:  # burasi 0.3du
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3) #Conf ile aldığım değeri (doğruluk değerini ekranda open cv aracılığıyla gösterir) ve ekranda gösterilen sayısal değerin sınırlarda kalmasını sağladım
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5) # oradaki l yeşil çerçeveyi biraz daha azaltır
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    resultsTracker = trackers.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)  # Line Red

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=10)  # Conf ile aldığım değeri (doğruluk değerini ekranda open cv aracılığıyla gösterir) ve ekranda gösterilen sayısal değerin sınırlarda kalmasını sağladım

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Line green
                totalCount.append(id)

    # cvzone.putTextRect(img, f'Toplam: {len(totalCount)}', (50,100))
    cv2.putText(img, str(len(totalCount)), (250, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion",imgRegion) #mask nasıl çalışıyor görmek için aç
    print(totalCount)

    cv2.waitKey(2)  # sadece 0 a bastığımda kare kare atlıyor
