# Smart Parking System - Hệ thống Bãi đỗ xe Thông minh

## Mô tả

Hệ thống quản lý bãi đỗ xe tự động với các tính năng:
- Nhận diện biển số xe tự động (YOLOv8 + EasyOCR)
- Quản lý ra/vào bằng thẻ RFID
- Thanh toán điện tử qua VietQR
- Giám sát thời gian thực qua Desktop App và Web Dashboard

## Cấu trúc thư mục

```
datnh/
├── project/          # Server Application (Python)
│   ├── parking_ui.py      # Desktop App chính (PySide6)
│   ├── run.py             # Auto-reload runner
│   ├── sepay_config.py    # Cấu hình thanh toán SePay
│   ├── sepay_helper.py    # Helper functions cho SePay
│   ├── qr_payment_widget.py
│   ├── requirements.txt   # Python dependencies
│   └── docs/              # Tài liệu kỹ thuật
│
├── thanhtoan/        # ESP32 Payment Terminal (C++)
│   ├── main.cpp           # Code chính cho gate02
│   ├── ota.h              # OTA update support
│   └── platformio.ini     # PlatformIO config
│
└── baidoxe/          # ESP32 Entry/Exit Gate (C++)
    ├── main.cpp           # Code chính cho gate01
    ├── ota.h              # OTA update support
    └── platformio.ini     # PlatformIO config
```

## Công nghệ sử dụng

### Server (project/)
| Component | Technology |
|-----------|------------|
| GUI Framework | PySide6 (Qt6) |
| Web Framework | Flask |
| AI Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| Database | PostgreSQL |
| IoT Protocol | MQTT (paho-mqtt) |
| Payment | VietQR + SePay API |

### ESP32 (thanhtoan/, baidoxe/)
| Component | Technology |
|-----------|------------|
| MCU | ESP32-S3 |
| Framework | Arduino (PlatformIO) |
| WiFi | WiFiManager |
| MQTT | PubSubClient |
| RFID | MFRC522 |
| Display | ST7735 TFT |

## Cài đặt

### Server (Python)
```bash
cd project
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python run.py
```

### ESP32 (PlatformIO)
```bash
cd thanhtoan  # hoặc baidoxe
pio run --target upload
```

## Tác giả

**Nguyễn Tuấn Anh**

## License

MIT License
