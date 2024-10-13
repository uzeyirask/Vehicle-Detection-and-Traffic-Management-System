# Vehicle Detection and Traffic Management System

This project utilizes YOLO and SORT algorithms for vehicle detection and tracking in traffic management. It analyzes camera footage to classify vehicles, assigns unique IDs to each detected vehicle, and stores the vehicle counts in an MSSQL database. The real-time traffic density is visualized on the screen. 

The main goal of this project is to enable real-time vehicle detection and counting to analyze traffic density. The system processes camera footage, detects vehicles, assigns unique IDs to each vehicle, and stores the data in an MSSQL database. While no in-depth analysis is performed in this version, the collected data has significant potential for further traffic analysis and optimization. The stored data can be used to develop more efficient traffic management strategies. Additionally, visualizing traffic density helps road users make informed decisions and contributes to better urban traffic management.


# Features
- **Vehicle Detection**: Uses YOLO for detecting vehicles in real-time.
- **Object Tracking**: Implements SORT algorithm for tracking vehicles across frames.
- **Real-Time Vehicle Counting**: Displays and updates vehicle count live.
- **Traffic Density Visualization**: Graphically shows traffic density.
- **MSSQL Database Integration**: Stores vehicle count and timestamp in an MSSQL database.
- **User-Friendly Interface**: Easy-to-understand visual outputs and real-time data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/uzeyirask/Vehicle-Detection-and-Traffic-Management-System.git
   
pip install -r requirements.txt

# Requirements

- **Python 3.10**
- **PyTorch with CUDA support (recommended)**
- **OpenCV**
- **PyODBC (for MSSQL database interaction)**
- **Ultralytics YOLO (for vehicle detection)**
- **SORT (for object tracking)**
- **MSSQL Server (for vehicle data storage)**

## FPS Optimization with CUDA
This project uses CUDA to leverage GPU acceleration, ensuring optimal frame rate for real-time vehicle detection.

## Screenshots
Here are screenshots showing the system in action


Sekil 1: Windows Forms (WinForms) App
![Windows Forms (WinForms) App](https://github.com/user-attachments/assets/ebaaf083-d425-4420-a0e9-51c6641d588e)


Sekil 2: Kamera-Yol Adres Bilgisi
![uygArayuz2](https://github.com/user-attachments/assets/5b6348be-a715-4919-9c00-1aa42fb725f2)


Sekil 3: Proje Adımları
![KULLANIMI](https://github.com/user-attachments/assets/a726254b-382e-43a1-a701-d5ccf70e8d96)

Sekil 4: Proje Adımları
![surec1](https://github.com/user-attachments/assets/459031f9-89d3-406a-891c-56f487134fc4)

Sekil 5: Çizilen Kutular ve Id Dizisi
![WhatInTheBox](https://github.com/user-attachments/assets/c1283b71-b4e9-4599-83a6-5888560bd769)

Sekil 6: Yogun Trafik
![yogunluk](https://github.com/user-attachments/assets/2dfb61cc-b893-473e-b62e-2879339c029d)

Sekil 7: Kutu Çizimi
![50car](https://github.com/user-attachments/assets/93253d87-0a5e-4814-bdad-fe10a8fc9fc9)

Sekil 8: Veri Tabani Yapısı
![veriTabani](https://github.com/user-attachments/assets/a8b7889b-2b8f-4603-906c-c14c39b8302e)
