# BÁO CÁO CÔNG NGHỆ HỆ THỐNG BÃI ĐỖ XE THÔNG MINH
## Smart Parking System - Technical Documentation

**Phiên bản:** 1.0
**Ngày cập nhật:** 24/11/2024
**Tác giả:** Nguyễn Tuấn Anh

---

## MỤC LỤC

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc Server](#2-kiến-trúc-server)
3. [Công nghệ phần mềm](#3-công-nghệ-phần-mềm)
4. [Giao thức truyền thông](#4-giao-thức-truyền-thông)
5. [Cơ sở dữ liệu](#5-cơ-sở-dữ-liệu)
6. [Trí tuệ nhân tạo (AI/ML)](#6-trí-tuệ-nhân-tạo-aiml)
7. [Phần cứng IoT](#7-phần-cứng-iot)
8. [Tích hợp thanh toán](#8-tích-hợp-thanh-toán)
9. [Bảo mật](#9-bảo-mật)
10. [Triển khai & Vận hành](#10-triển-khai--vận-hành)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Mô tả chung

Hệ thống Bãi đỗ xe Thông minh (Smart Parking System) là giải pháp tự động hóa hoàn toàn quy trình quản lý bãi đỗ xe, bao gồm:

- **Nhận diện biển số tự động** bằng AI (YOLOv8 + EasyOCR)
- **Quản lý ra/vào** bằng thẻ RFID
- **Thanh toán điện tử** qua VietQR/Banking App
- **Giám sát thời gian thực** qua Web và Desktop App

### 1.2. Kiến trúc 4 lớp (4-Layer Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KIẾN TRÚC HỆ THỐNG                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LỚP 4: GIAO DIỆN NGƯỜI DÙNG (Presentation Layer)               │   │
│  │    • Desktop App (PySide6/Qt6)                                   │   │
│  │    • Web Dashboard (Flask + Jinja2)                              │   │
│  │    • Mobile Banking App (thanh toán)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ HTTP/WebSocket                           │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LỚP 3: XỬ LÝ NGHIỆP VỤ (Business Logic Layer)                  │   │
│  │    • Flask Backend Server                                        │   │
│  │    • AI Engine (YOLOv8 + EasyOCR)                                │   │
│  │    • Payment Processing (SePay API)                              │   │
│  │    • MQTT Message Broker                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ SQL/ORM                                  │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LỚP 2: DỮ LIỆU (Data Layer)                                    │   │
│  │    • PostgreSQL Database                                         │   │
│  │    • File Storage (ảnh biển số)                                  │   │
│  │    • Session Cache                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ MQTT/SPI                                 │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LỚP 1: PHẦN CỨNG IoT (Hardware Layer)                          │   │
│  │    • ESP32-S3 Controllers (2 units)                              │   │
│  │    • RFID RC522 Readers                                          │   │
│  │    • USB Camera                                                  │   │
│  │    • TFT Display (ST7735)                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3. Sơ đồ luồng dữ liệu

```
                    ┌─────────────┐
                    │   CAMERA    │
                    │  (USB/IP)   │
                    └──────┬──────┘
                           │ Video Stream
                           ▼
┌──────────┐        ┌─────────────┐        ┌─────────────┐
│  RFID    │───────▶│   DESKTOP   │◀──────▶│  POSTGRESQL │
│  RC522   │  MQTT  │     APP     │  SQL   │  DATABASE   │
└──────────┘        │  (Python)   │        └─────────────┘
     │              └──────┬──────┘               ▲
     │                     │                      │
     ▼                     ▼                      │
┌──────────┐        ┌─────────────┐        ┌─────────────┐
│  ESP32   │◀──────▶│    MQTT     │◀──────▶│    FLASK    │
│  (IoT)   │  MQTT  │   BROKER    │  MQTT  │   WEB APP   │
└──────────┘        └─────────────┘        └─────────────┘
     │                     ▲                      │
     ▼                     │                      ▼
┌──────────┐               │              ┌─────────────┐
│   TFT    │               │              │   BROWSER   │
│ DISPLAY  │               │              │  (Web GUI)  │
└──────────┘               │              └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   SEPAY     │
                    │    API      │
                    └─────────────┘
```

---

## 2. KIẾN TRÚC SERVER

### 2.1. Thông số Server

| Thành phần | Thông số kỹ thuật |
|------------|-------------------|
| **Hệ điều hành** | Windows 11 / Ubuntu Server 22.04 |
| **CPU** | Intel Core i5/i7 hoặc AMD Ryzen 5/7 (khuyến nghị) |
| **RAM** | Tối thiểu 8GB (khuyến nghị 16GB cho AI) |
| **GPU** | NVIDIA GTX 1650+ với CUDA 12.1 (cho YOLOv8) |
| **Storage** | SSD 256GB+ (cho tốc độ đọc/ghi) |
| **Network** | Gigabit Ethernet / WiFi 5GHz |

### 2.2. Các thành phần Server

#### 2.2.1. Application Server

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION SERVER                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    PYTHON RUNTIME (3.11+)                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │
│  │  │  Desktop    │  │   Flask     │  │    AI       │               │ │
│  │  │    App      │  │  Web App    │  │   Engine    │               │ │
│  │  │  (PySide6)  │  │  (REST API) │  │  (PyTorch)  │               │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │ │
│  │         │                │                │                       │ │
│  │         └────────────────┼────────────────┘                       │ │
│  │                          │                                        │ │
│  │                          ▼                                        │ │
│  │              ┌─────────────────────┐                              │ │
│  │              │   Shared Libraries  │                              │ │
│  │              │  • psycopg2 (DB)    │                              │ │
│  │              │  • paho-mqtt        │                              │ │
│  │              │  • requests (HTTP)  │                              │ │
│  │              └─────────────────────┘                              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Port Allocation:                                                       │
│    • 5000: Flask Web Application                                        │
│    • 5432: PostgreSQL Database                                          │
│    • 1883: MQTT Broker (Mosquitto)                                      │
│    • 8883: MQTT over TLS (optional)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.2. Database Server

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        POSTGRESQL SERVER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Version: PostgreSQL 14+                                                │
│  Port: 5432                                                             │
│  Encoding: UTF-8                                                        │
│  Collation: Vietnamese_Vietnam.1258                                     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      DATABASE: parking_db                         │ │
│  │                                                                   │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │ parking_sessions│  │   rfid_cards    │  │     users       │   │ │
│  │  │  • id           │  │  • id           │  │  • id           │   │ │
│  │  │  • card_id      │  │  • card_uid     │  │  • username     │   │ │
│  │  │  • plate_number │  │  • owner_name   │  │  • password     │   │ │
│  │  │  • time_in      │  │  • vehicle_type │  │  • role         │   │ │
│  │  │  • time_out     │  │  • is_active    │  │  • created_at   │   │ │
│  │  │  • fee          │  │  • created_at   │  │                 │   │ │
│  │  │  • paid         │  └─────────────────┘  └─────────────────┘   │ │
│  │  │  • image_in     │                                             │ │
│  │  │  • image_out    │  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  └─────────────────┘  │    payments     │  │   audit_logs    │   │ │
│  │                       │  • id           │  │  • id           │   │ │
│  │                       │  • session_id   │  │  • action       │   │ │
│  │                       │  • amount       │  │  • user_id      │   │ │
│  │                       │  • method       │  │  • timestamp    │   │ │
│  │                       │  • transaction  │  │  • details      │   │ │
│  │                       └─────────────────┘  └─────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Connection Pooling: SQLAlchemy (pool_size=10, max_overflow=20)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.3. MQTT Broker Server

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MQTT BROKER (MOSQUITTO)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Host: linuxtuananh.zapto.org (DDNS)                                    │
│  Port: 1883 (TCP) / 8883 (TLS)                                          │
│  Protocol: MQTT 3.1.1 / 5.0                                             │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      TOPIC HIERARCHY                              │ │
│  │                                                                   │ │
│  │  parking/                                                         │ │
│  │    ├── gate/                                                      │ │
│  │    │     ├── gate01/                    (Entry/Exit Gate)         │ │
│  │    │     │     ├── heartbeat            QoS 0, 4s interval        │ │
│  │    │     │     ├── status               QoS 1, retained           │ │
│  │    │     │     ├── card_scanned         QoS 1                     │ │
│  │    │     │     └── take_photo           QoS 1                     │ │
│  │    │     └── gate02/                    (Payment Terminal)        │ │
│  │    │           └── ...                                            │ │
│  │    │                                                              │ │
│  │    └── payment/                                                   │ │
│  │          └── gate02/                                              │ │
│  │                ├── vehicle_info         QoS 1 (JSON payload)      │ │
│  │                ├── qr_data              QoS 1 (Base64 PNG)        │ │
│  │                ├── payment_confirmed    QoS 1                     │ │
│  │                └── add_card_mode        QoS 1                     │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  QoS Levels:                                                            │
│    • QoS 0: At most once (heartbeat, non-critical)                      │
│    • QoS 1: At least once (card scan, payment)                          │
│    • QoS 2: Exactly once (not used - overhead)                          │
│                                                                         │
│  Features:                                                              │
│    • Retained messages for status                                       │
│    • Last Will Testament (LWT) for offline detection                    │
│    • Clean session = false (persistent subscriptions)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3. Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NETWORK TOPOLOGY                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌──────────────┐                                │
│                         │   INTERNET   │                                │
│                         └──────┬───────┘                                │
│                                │                                        │
│                         ┌──────▼───────┐                                │
│                         │    ROUTER    │                                │
│                         │  (NAT/DDNS)  │                                │
│                         └──────┬───────┘                                │
│                                │ 192.168.1.1                            │
│              ┌─────────────────┼─────────────────┐                      │
│              │                 │                 │                      │
│       ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐              │
│       │   SERVER    │   │   ESP32     │   │   ESP32     │              │
│       │  (Desktop)  │   │  (gate01)   │   │  (gate02)   │              │
│       │192.168.1.100│   │192.168.1.60 │   │192.168.1.61 │              │
│       └──────┬──────┘   └─────────────┘   └─────────────┘              │
│              │                                                          │
│       ┌──────▼──────┐                                                   │
│       │   CAMERA    │                                                   │
│       │   (USB)     │                                                   │
│       └─────────────┘                                                   │
│                                                                         │
│  Port Forwarding (for remote access):                                   │
│    • 1883 → 192.168.1.100:1883 (MQTT)                                   │
│    • 5000 → 192.168.1.100:5000 (Flask Web)                              │
│    • 22   → 192.168.1.100:22   (SSH)                                    │
│                                                                         │
│  DDNS: linuxtuananh.zapto.org → Public IP                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4. Server Processes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RUNNING PROCESSES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Process Name          Port    CPU    RAM     Description               │
│  ─────────────────────────────────────────────────────────────────────  │
│  python parking_ui.py   -      15%    2GB     Desktop App + AI Engine   │
│  python flask_app.py   5000    5%     500MB   Web Application           │
│  mosquitto             1883    1%     50MB    MQTT Broker               │
│  postgresql            5432    3%     200MB   Database Server           │
│                                                                         │
│  Startup Order:                                                         │
│    1. PostgreSQL (database must be ready first)                         │
│    2. Mosquitto (MQTT broker for IoT)                                   │
│    3. Flask Web App (optional, for web dashboard)                       │
│    4. Desktop App (main application with AI)                            │
│                                                                         │
│  Auto-restart: run.py (watchdog for hot-reload during development)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. CÔNG NGHỆ PHẦN MỀM

### 3.1. Ngôn ngữ lập trình

| Ngôn ngữ | Phiên bản | Mục đích sử dụng |
|----------|-----------|------------------|
| **Python** | 3.11+ | Backend, AI/ML, Desktop App |
| **C/C++** | C++17 (Arduino) | Firmware ESP32 |
| **SQL** | PostgreSQL 14+ | Database queries |
| **HTML5** | - | Web structure |
| **CSS3** | - | Web styling |
| **JavaScript** | ES6+ | Web interactivity |

### 3.2. Python Frameworks & Libraries

#### 3.2.1. Desktop Application (GUI)

| Library | Version | Description |
|---------|---------|-------------|
| **PySide6** | 6.9.2 | Qt6 binding - Cross-platform GUI framework |
| **Qt6** | 6.9 | Native UI components, signals/slots |

**Các thành phần Qt6 sử dụng:**
- `QMainWindow`: Cửa sổ chính
- `QWidget`: Base widget class
- `QLabel`: Hiển thị text/image
- `QTimer`: Polling, animation
- `QThread`: Multi-threading
- `Signal/Slot`: Event handling

#### 3.2.2. Web Framework

| Library | Version | Description |
|---------|---------|-------------|
| **Flask** | 3.1.2 | Micro web framework |
| **Flask-SocketIO** | 5.5.1 | Real-time WebSocket |
| **Flask-SQLAlchemy** | 3.1.1 | ORM integration |
| **Flask-Login** | 0.6.3 | User session management |
| **Flask-Security-Too** | 5.6.2 | Authentication & Authorization |
| **Flask-Migrate** | 4.1.0 | Database migrations |
| **Flask-WTF** | 1.2.2 | Form handling + CSRF |
| **Flask-Compress** | 1.20 | Response compression |
| **Jinja2** | 3.1.6 | Template engine |
| **Werkzeug** | 3.1.3 | WSGI utilities |

#### 3.2.3. AI/Machine Learning

| Library | Version | Description |
|---------|---------|-------------|
| **Ultralytics** | 8.3.203 | YOLOv8 object detection |
| **EasyOCR** | 1.7.2 | Optical Character Recognition |
| **PyTorch** | 2.5.1+cu121 | Deep Learning framework |
| **TorchVision** | 0.20.1+cu121 | Computer vision utilities |
| **OpenCV** | 4.12.0 | Image/video processing |
| **NumPy** | 2.2.6 | Numerical computing |
| **Pillow** | 11.3.0 | Image manipulation |
| **scikit-image** | 0.25.2 | Image processing algorithms |

#### 3.2.4. Database

| Library | Version | Description |
|---------|---------|-------------|
| **psycopg2** | 2.9.11 | PostgreSQL adapter (sync) |
| **psycopg** | 3.2.10 | PostgreSQL adapter (async) |
| **SQLAlchemy** | 2.0.44 | ORM (Object Relational Mapper) |
| **Alembic** | 1.17.0 | Database migration tool |

#### 3.2.5. IoT & Communication

| Library | Version | Description |
|---------|---------|-------------|
| **paho-mqtt** | 2.1.0 | MQTT client library |
| **python-socketio** | 5.14.2 | Socket.IO client |
| **requests** | 2.32.5 | HTTP client |
| **paramiko** | 3.5.1 | SSH client (remote access) |
| **sshtunnel** | 0.4.0 | SSH tunneling |

#### 3.2.6. Utilities

| Library | Version | Description |
|---------|---------|-------------|
| **qrcode** | 8.2 | QR code generation |
| **watchdog** | 6.0.0 | File system monitoring |
| **colorama** | 0.4.6 | Terminal colors |
| **PyYAML** | 6.0.3 | YAML parsing |
| **python-dateutil** | 2.9.0 | Date/time utilities |

### 3.3. ESP32 Libraries (PlatformIO)

| Library | Description |
|---------|-------------|
| **WiFi.h** | ESP32 WiFi connectivity |
| **WiFiManager** | Captive portal for WiFi setup |
| **PubSubClient** | MQTT client |
| **ArduinoJson** | JSON serialization/deserialization |
| **MFRC522** | RFID RC522 driver |
| **Adafruit_ST7735** | TFT display driver |
| **Adafruit_GFX** | Graphics library |
| **ArduinoOTA** | Over-the-Air updates |
| **FreeRTOS** | Real-time operating system |

---

## 4. GIAO THỨC TRUYỀN THÔNG

### 4.1. MQTT (Message Queuing Telemetry Transport)

#### 4.1.1. Đặc điểm kỹ thuật

| Thuộc tính | Giá trị |
|------------|---------|
| **Protocol Version** | MQTT 3.1.1 |
| **Transport** | TCP/IP |
| **Port** | 1883 (plaintext), 8883 (TLS) |
| **Broker** | Eclipse Mosquitto |
| **Message Format** | JSON |
| **Max Payload** | 256 MB (configured: 1 MB) |

#### 4.1.2. Topic Structure

```
parking/
├── gate/
│   ├── {gate_id}/
│   │   ├── heartbeat          # Tín hiệu sống (mỗi 4 giây)
│   │   ├── status             # Trạng thái online/offline
│   │   ├── card_scanned       # Thẻ RFID được quét
│   │   └── take_photo         # Yêu cầu chụp ảnh
│   │
├── payment/
│   └── {gate_id}/
│       ├── vehicle_info       # Thông tin xe (biển số, phí)
│       ├── qr_data            # Mã QR thanh toán
│       ├── payment_confirmed  # Xác nhận thanh toán
│       └── add_card_mode      # Chế độ thêm thẻ mới
```

#### 4.1.3. Message Examples

**Heartbeat Message:**
```json
{
    "time": 123456789,
    "mac": "AA:BB:CC:DD:EE:FF",
    "ip": "192.168.1.60"
}
```

**Card Scanned Message:**
```json
{
    "card_id": "A1B2C3D4",
    "reader": "IN",
    "timestamp": "2024-11-24T10:30:00Z"
}
```

**Vehicle Info Message:**
```json
{
    "plate": "59A-12345",
    "time_in": "10:30:00",
    "time_out": "14:30:00",
    "fee": 20000,
    "paid": false
}
```

**Payment Confirmed Message:**
```json
{
    "session_id": 123,
    "status": "paid",
    "amount": 20000,
    "card_id": "A1B2C3D4"
}
```

### 4.2. HTTP/HTTPS REST API

#### 4.2.1. Internal API (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | Lấy danh sách phiên gửi xe |
| `/api/sessions/<id>` | GET | Chi tiết một phiên |
| `/api/sessions/<id>/checkout` | POST | Xử lý thanh toán |
| `/api/cards` | GET/POST | Quản lý thẻ RFID |
| `/api/stats` | GET | Thống kê bãi xe |

#### 4.2.2. External API (SePay/VietQR)

**VietQR API:**
```
POST https://api.vietqr.io/v2/generate
Content-Type: application/json

{
    "accountNo": "102874512400",
    "accountName": "NGUYEN TUAN ANH",
    "acqId": "970415",
    "amount": 20000,
    "addInfo": "SEVQR BAI XE SESSION 123",
    "format": "text",
    "template": "compact"
}
```

**SePay API:**
```
GET https://my.sepay.vn/userapi/transactions/list
Authorization: Bearer {API_TOKEN}

Query params:
  - limit: 10
  - amount_in: 20000
```

### 4.3. WebSocket (Socket.IO)

#### 4.3.1. Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Kết nối mới |
| `disconnect` | Client → Server | Ngắt kết nối |
| `vehicle_entry` | Server → Client | Xe vào bãi |
| `vehicle_exit` | Server → Client | Xe ra bãi |
| `payment_update` | Server → Client | Cập nhật thanh toán |
| `dashboard_refresh` | Server → Client | Làm mới dashboard |

### 4.4. SPI (Serial Peripheral Interface)

#### 4.4.1. Cấu hình ESP32

```cpp
// SPI Pin Configuration
#define SPI_MOSI  11
#define SPI_MISO  13
#define SPI_SCK   12

// Device Chip Select pins
#define RFID_CS   14  // RC522
#define TFT_CS    10  // ST7735
```

#### 4.4.2. Thông số

| Device | Clock Speed | Mode | Data Order |
|--------|-------------|------|------------|
| RFID RC522 | 4 MHz | Mode 0 | MSB First |
| TFT ST7735 | 20 MHz | Mode 0 | MSB First |

---

## 5. CƠ SỞ DỮ LIỆU

### 5.1. PostgreSQL Configuration

| Parameter | Value |
|-----------|-------|
| **Version** | 14+ |
| **Port** | 5432 |
| **Encoding** | UTF-8 |
| **Max Connections** | 100 |
| **Shared Buffers** | 256 MB |

### 5.2. Database Schema

#### 5.2.1. Bảng `parking_sessions`

```sql
CREATE TABLE parking_sessions (
    id SERIAL PRIMARY KEY,
    card_id VARCHAR(20) NOT NULL,
    plate_number VARCHAR(20),
    time_in TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    time_out TIMESTAMP WITH TIME ZONE,
    fee DECIMAL(10,2) DEFAULT 0,
    paid BOOLEAN DEFAULT FALSE,
    image_in TEXT,          -- Base64 encoded
    image_out TEXT,         -- Base64 encoded
    gate_in VARCHAR(10),
    gate_out VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_sessions_card_id ON parking_sessions(card_id);
CREATE INDEX idx_sessions_time_in ON parking_sessions(time_in);
CREATE INDEX idx_sessions_paid ON parking_sessions(paid);
```

#### 5.2.2. Bảng `rfid_cards`

```sql
CREATE TABLE rfid_cards (
    id SERIAL PRIMARY KEY,
    card_uid VARCHAR(20) UNIQUE NOT NULL,
    owner_name VARCHAR(100),
    phone VARCHAR(20),
    vehicle_type VARCHAR(20) DEFAULT 'car',
    plate_number VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    balance DECIMAL(12,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 5.2.3. Bảng `users`

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'operator',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5.3. Tính phí tự động

```sql
-- Công thức tính phí trong SQL
SELECT
    id,
    card_id,
    plate_number,
    time_in,
    CASE
        WHEN time_out IS NULL THEN NOW()
        ELSE time_out
    END as calc_time_out,
    CASE
        WHEN time_out IS NULL THEN
            3000 + (FLOOR(EXTRACT(EPOCH FROM (NOW() - time_in)) / 3600) * 5000)
        ELSE COALESCE(fee, 0)
    END as calc_fee
FROM parking_sessions
WHERE paid = FALSE;

-- Giải thích:
-- Phí cơ bản: 3,000 VND
-- Phí theo giờ: 5,000 VND/giờ
```

---

## 6. TRÍ TUỆ NHÂN TẠO (AI/ML)

### 6.1. License Plate Recognition Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI PROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Camera  │───▶│ Preprocessing│───▶│   YOLOv8   │───▶│  Crop ROI  │  │
│  │  Frame  │    │  (OpenCV)   │    │  Detection │    │            │  │
│  └─────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                                              │         │
│  Input: 1920x1080                                            ▼         │
│                                                       ┌─────────────┐  │
│  ┌─────────────┐    ┌─────────────┐                  │  EasyOCR   │  │
│  │   Output    │◀───│ Post-process│◀─────────────────│ Recognition│  │
│  │  "59A-123"  │    │  (Cleanup)  │                  └─────────────┘  │
│  └─────────────┘    └─────────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2. YOLOv8 Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv8n (nano) / YOLOv8s (small) |
| **Input Size** | 640 x 640 pixels |
| **Classes** | 1 (license_plate) |
| **Confidence Threshold** | 0.5 |
| **IOU Threshold** | 0.45 |
| **Device** | CUDA (GPU) / CPU fallback |

**Model file:** `license_plate_detector.pt`

### 6.3. EasyOCR Configuration

```python
# EasyOCR initialization
reader = easyocr.Reader(
    ['vi', 'en'],           # Languages: Vietnamese, English
    gpu=True,               # Use GPU if available
    model_storage_directory='models/',
    download_enabled=True
)

# Character whitelist for license plates
allowlist = '0123456789ABCDEFGHKLMNPSTUVXYZ-.'
```

### 6.4. Performance Metrics

| Metric | Value |
|--------|-------|
| **Detection FPS** | 30+ (GPU) / 5-10 (CPU) |
| **OCR Accuracy** | 95%+ |
| **Total Latency** | < 500ms |
| **GPU Memory** | ~2 GB |

---

## 7. PHẦN CỨNG IoT

### 7.1. ESP32-S3 Specifications

| Parameter | Value |
|-----------|-------|
| **Chip** | ESP32-S3 (Dual-core Xtensa LX7) |
| **Clock** | 240 MHz |
| **RAM** | 512 KB SRAM + 8 MB PSRAM |
| **Flash** | 16 MB |
| **WiFi** | 802.11 b/g/n (2.4 GHz) |
| **Bluetooth** | BLE 5.0 |

### 7.2. Hardware Components

#### 7.2.1. GATE01 (Entry/Exit Gate)

| Component | Model | Quantity | Function |
|-----------|-------|----------|----------|
| MCU | ESP32-S3 | 1 | Main controller |
| RFID Reader | MFRC522 | 2 | IN/OUT card reader |
| Relay | 5V Relay | 2 | Gate barrier control |

#### 7.2.2. GATE02 (Payment Terminal)

| Component | Model | Quantity | Function |
|-----------|-------|----------|----------|
| MCU | ESP32-S3 | 1 | Main controller |
| RFID Reader | MFRC522 | 1 | Card reader |
| TFT Display | ST7735 128x160 | 1 | QR code display |

### 7.3. Pin Configuration

```
ESP32-S3 GPIO Mapping:
─────────────────────────────────────────
│ GPIO │ Function        │ Device        │
─────────────────────────────────────────
│  11  │ SPI MOSI        │ RFID/TFT      │
│  12  │ SPI SCK         │ RFID/TFT      │
│  13  │ SPI MISO        │ RFID          │
│  14  │ CS (RFID A)     │ MFRC522 IN    │
│  15  │ RST (RFID A)    │ MFRC522 IN    │
│  10  │ CS (RFID B/TFT) │ MFRC522 OUT   │
│   9  │ RST (RFID B)    │ MFRC522 OUT   │
│   6  │ DC              │ ST7735        │
│   7  │ RST             │ ST7735        │
─────────────────────────────────────────
```

---

## 8. TÍCH HỢP THANH TOÁN

### 8.1. VietQR Standard

| Parameter | Value |
|-----------|-------|
| **Standard** | VietQR (NAPAS) |
| **Bank** | VietinBank |
| **ACQ ID** | 970415 |
| **Account** | 102874512400 |
| **Format** | EMVCo QR |

### 8.2. Payment Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PAYMENT WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. User scans RFID card at exit                                        │
│         │                                                               │
│         ▼                                                               │
│  2. System calculates fee and generates VietQR                          │
│         │                                                               │
│         ▼                                                               │
│  3. QR displayed on TFT screen                                          │
│         │                                                               │
│         ▼                                                               │
│  4. User scans QR with banking app                                      │
│         │                                                               │
│         ▼                                                               │
│  5. User confirms payment in banking app                                │
│         │                                                               │
│         ▼                                                               │
│  6. Bank transfers money to parking account                             │
│         │                                                               │
│         ▼                                                               │
│  7. SePay detects transaction                                           │
│         │                                                               │
│         ├──▶ Option A: Webhook notification                             │
│         │                                                               │
│         └──▶ Option B: API polling (every 5 seconds)                    │
│                  │                                                      │
│                  ▼                                                      │
│  8. System verifies: amount + description match                         │
│         │                                                               │
│         ▼                                                               │
│  9. Payment confirmed → Gate opens                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3. QR Content Format

```
Nội dung chuyển khoản: SEVQR BAI XE SESSION {session_id}

Ví dụ: SEVQR BAI XE SESSION 123

Giải thích:
- SEVQR: Prefix bắt buộc cho SePay + VietinBank
- BAI XE: Identifier hệ thống
- SESSION: Từ khóa
- {session_id}: ID phiên gửi xe (unique)
```

---

## 9. BẢO MẬT

### 9.1. Authentication & Authorization

| Layer | Method |
|-------|--------|
| **Web App** | Flask-Login + Flask-Security |
| **API** | Bearer Token (SePay) |
| **MQTT** | Username/Password (optional) |
| **Database** | PostgreSQL roles |

### 9.2. Security Features

| Feature | Implementation |
|---------|----------------|
| **Password Hashing** | bcrypt (Passlib) |
| **CSRF Protection** | Flask-WTF tokens |
| **Session Management** | Secure cookies |
| **SQL Injection** | SQLAlchemy ORM (parameterized) |
| **XSS Prevention** | Jinja2 auto-escaping |

### 9.3. ESP32 Security

| Feature | Implementation |
|---------|----------------|
| **OTA Updates** | Password protected |
| **WiFi Config** | WPA2 + Captive Portal |
| **MQTT** | Optional TLS (port 8883) |

---

## 10. TRIỂN KHAI & VẬN HÀNH

### 10.1. Development Environment

```bash
# Clone project
git clone <repository>
cd project

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start services
# 1. PostgreSQL (already running)
# 2. Mosquitto MQTT broker
mosquitto -v

# 3. Flask Web App (optional)
python flask_app.py

# 4. Desktop App with hot-reload
python run.py
```

### 10.2. Production Deployment

```
Production Stack:
├── Gunicorn (WSGI server for Flask)
├── Nginx (Reverse proxy, SSL termination)
├── PostgreSQL (Production database)
├── Mosquitto (MQTT broker with auth)
├── Systemd (Service management)
└── Certbot (Let's Encrypt SSL)
```

### 10.3. Monitoring

| Tool | Purpose |
|------|---------|
| **Logging** | Python logging module |
| **MQTT Monitor** | MQTT Explorer / mosquitto_sub |
| **Database** | pgAdmin 4 |
| **Network** | Wireshark |

---

## PHỤ LỤC

### A. Danh sách thư viện đầy đủ (requirements.txt)

Xem file `requirements.txt` trong thư mục gốc dự án.

### B. Cấu trúc thư mục dự án

```
project/
├── parking_ui.py          # Desktop Application (main)
├── run.py                 # Development runner with hot-reload
├── sepay_config.py        # Payment configuration
├── sepay_helper.py        # Payment utilities
├── qr_payment_widget.py   # QR display widget
├── requirements.txt       # Python dependencies
├── config.json            # Application configuration
├── plates/                # License plate images
│   ├── IN/               # Entry images
│   └── OUT/              # Exit images
├── models/               # AI models
│   └── license_plate_detector.pt
├── templates/            # Flask HTML templates
├── static/               # Static files (CSS, JS)
└── docs/                 # Documentation
    ├── BAO_CAO_CONG_NGHE.md
    └── system_diagrams/
```

### C. Tài liệu tham khảo

1. Flask Documentation: https://flask.palletsprojects.com/
2. PySide6 Documentation: https://doc.qt.io/qtforpython/
3. YOLOv8 (Ultralytics): https://docs.ultralytics.com/
4. EasyOCR: https://github.com/JaidedAI/EasyOCR
5. Paho MQTT: https://eclipse.dev/paho/
6. PostgreSQL: https://www.postgresql.org/docs/
7. VietQR: https://vietqr.io/
8. SePay API: https://my.sepay.vn/

---

**© 2024 - Hệ thống Bãi đỗ xe Thông minh**
