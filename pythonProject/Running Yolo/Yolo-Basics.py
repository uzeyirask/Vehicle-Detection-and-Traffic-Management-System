#from ultralytics import YOLO
#import cv2
#
#model = YOLO('../Yolo-Weights/yolov8l.pt')
#result = model("Images/1.png", show=True)
#cv2.waitKey(0)


from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
image_path = "Images/2.png"
image = cv2.imread(image_path)

# Resmi işle
results = model(image)

# İşlenmiş sonuçları al
processed_results = results[0].cpu().numpy()

# Sonuçları ekranda göster
for detection in processed_results:
    if len(detection) < 6:  # Algılamada sınırlayıcı kutu yoksa
        continue  # Geç ve bir sonraki algılamaya geç

    class_id = int(detection[5])
    label = model.names[class_id]
    confidence = detection[4]
    x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Ekranda göster
cv2.imshow('Results', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
