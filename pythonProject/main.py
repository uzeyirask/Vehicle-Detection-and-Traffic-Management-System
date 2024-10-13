import torch
import cv2
from ultralytics import YOLO

# GPU kullanılabilir mi diye kontrol et
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Video yakalama nesnesini başlat
cap = cv2.VideoCapture("../Videos/cars.mp4")

# Modeli oluşturun
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Modeli CUDA cihazına taşıyın (sadece eğer CUDA kullanılabilirse)
if torch.cuda.is_available():
    model.cuda()

# Döngü içindeki işlemler
while True:
    success, img = cap.read()
    if not success:
        break  # Video bittiğinde döngüyü sonlandır
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV görüntü formatını PyTorch formatına dönüştürme
    img = torch.from_numpy(img).to(device)  # Görüntüyü CUDA cihazına taşı

    result = model(img, stream=True)

    # Geri kalan kod burada devam eder...

from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

#webCam icin kullanilan lar asagidaki
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

 # cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO('../Yolo-Weights/yolov8l.pt')  # Hangi yolo modelini kullacagimi buradan sectim

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella"
"handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat"
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli"
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"
]

while True:  # Cerceveler icin gerekli islemleri yapiyuoruz cerceve sekli cerceve rengi gibi
    success, img =cap.read()
    result =model(img,stream=True)
    for r in result:
        boxes =r.boxes
        for box in boxes:
            #Kutulari olusturma
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h =x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            #Kutu sinirlari ve fotograftakı nesnelerın dogruluk degerlerı
            conf = math.ceil((box.conf[0]*100))/100 #Ekranda gosterilen nesnelerin
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1) #Conf ile aldigim degeri(dogruluk degerini ekranda open cv araciligiyla gosterir) ve ekranda gosterilen sayisal degerin sinirlarda kalmasini sagladim

            #ClassName
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1) #Conf ile aldigim degeri(dogruluk degerini ekranda open cv araciligiyla gosterir) ve ekranda gosterilen sayisal degerin sinirlarda kalmasini sagladim

    cv2.imshow("Image",img)
    cv2.waitKey(1)