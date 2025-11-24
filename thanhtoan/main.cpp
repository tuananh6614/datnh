#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <qrcode.h>
#include <MFRC522.h>
#include <ArduinoOTA.h>
#include <HTTPClient.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "ota.h"  // OTA update support
#include <WiFiManager.h>  // ✅ WiFiManager for easy WiFi configuration

// ====== WIFI & MQTT CONFIG ======
// WiFi credentials now managed by WiFiManager (no hardcoded SSID/PASS)

const char* MQTT_HOST = "linuxtuananh.zapto.org";  // ✅ Updated to use domain name for remote access
const int   MQTT_PORT = 1883;
const char* GATE_ID   = "gate02";  // Payment Terminal

WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long lastHeartbeat = 0;
const unsigned long HEARTBEAT_INTERVAL = 4000;

// ====== TFT ST7735 CONFIG ======
#define TFT_CS    10
#define TFT_RST   7
#define TFT_DC    6
// TFT MOSI = 11, SCK = 12 (hardware SPI)

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// ====== LOG TAG ======
static const char* TAG = "PAYMENT_TERMINAL";

// ====== RFID RC522 CONFIG ======
// RFID Reader pins (giữ nguyên hardware)
#define RC522_MOSI  11
#define RC522_MISO  13
#define RC522_SCK   12
#define RC522_CS    14
#define RC522_RST   15

// MFRC522 instance
MFRC522 mfrc522(RC522_CS, RC522_RST);

// ====== HELPER FUNCTIONS ======
// Hàm format số tiền với dấu phẩy phân cách nghìn
void formatMoney(char* buffer, int amount) {
    if (amount < 1000) {
        sprintf(buffer, "%d", amount);
        return;
    }
    
    // Chuyển số thành chuỗi ngược
    char temp[20];
    sprintf(temp, "%d", amount);
    int len = strlen(temp);
    
    // Thêm dấu phẩy vào
    int j = 0;
    for (int i = len - 1, count = 0; i >= 0; i--, count++) {
        if (count == 3) {
            buffer[j++] = ',';
            count = 0;
        }
        buffer[j++] = temp[i];
    }
    buffer[j] = '\0';
    
    // Đảo ngược lại chuỗi
    int start = 0;
    int end = j - 1;
    while (start < end) {
        char t = buffer[start];
        buffer[start] = buffer[end];
        buffer[end] = t;
        start++;
        end--;
    }
}

// ====== STATE MACHINE ======
enum DisplayState {
    STATE_WAITING,          // Màn hình chờ
    STATE_SHOW_INFO,        // Hiển thị thông tin xe
    STATE_SHOW_QR,          // Hiển thị QR code để thanh toán
    STATE_PROCESSING,       // Animation xử lý thanh toán
    STATE_SHOW_PAID,        // Hiển thị "Đã thanh toán"
    STATE_SHOW_ERROR,       // Hiển thị lỗi
    STATE_ALREADY_PAID      // Thẻ đã thanh toán rồi
};

DisplayState currentState = STATE_WAITING;
unsigned long stateTimeout = 0;
const unsigned long INFO_TIMEOUT = 10000; // 10 giây - tự động thanh toán
int dots = 0;

// ====== RFID STATE ======
String lastCardUID = "";
unsigned long lastCardTime = 0;
const unsigned long CARD_DEBOUNCE = 3000; // 3 giây debounce

// ====== OTA STATE ======
bool isOTAUpdating = false;  // Flag để ngăn state machine vẽ đè lên màn hình OTA

// ====== VEHICLE INFO ======
struct VehicleInfo {
    String plate;
    String timeIn;
    String timeOut;
    int fee;
    bool hasData;
    bool isPaid;  // Trạng thái đã thanh toán
    String cardUID;  // Lưu card UID để request QR
} currentVehicle = {"", "", "", 0, false, false, ""};

// ====== ADD CARD MODE ======
bool addCardModeEnabled = false;  // Chế độ thêm thẻ mới
unsigned long addCardModeStartTime = 0;  // Thời gian bắt đầu chế độ thêm thẻ
const unsigned long ADD_CARD_TIMEOUT = 30000;  // 30 giây timeout

// ====== FORWARD DECLARATIONS ======
void displayAddCardMode();
void displayWelcomeScreen();
void displayVehicleInfo(bool fullRedraw = true);
void displayAlreadyPaid();
void displayLoadingScreen();
void displayPaymentProcessing();
void displayPaymentSuccess();
void displayPaymentError();

// ====== WIFI & MQTT FUNCTIONS ======
void wifiConnect() {
    Serial.println("[WIFI] Starting WiFiManager...");

    tft.fillScreen(ST77XX_BLACK);
    tft.setCursor(0, 10);
    tft.setTextColor(ST77XX_CYAN);
    tft.setTextSize(1);
    tft.println("WiFi Setup...");
    tft.println("");
    tft.setTextColor(ST77XX_YELLOW);
    tft.println("Neu chua co WiFi:");
    tft.println("1. Ket noi WiFi:");
    tft.setTextColor(ST77XX_GREEN);
    tft.println("  ESP32-ThanhToan");
    tft.setTextColor(ST77XX_YELLOW);
    tft.println("2. Mo trinh duyet:");
    tft.setTextColor(ST77XX_WHITE);
    tft.println("  192.168.4.1");

    WiFiManager wm;

    // Timeout 180 giây, sau đó restart
    wm.setConfigPortalTimeout(180);

    // Tự động kết nối hoặc mở config portal
    // AP Name: "ESP32-ThanhToan", Password: "12345678"
    bool connected = wm.autoConnect("ESP32-ThanhToan", "12345678");

    if (!connected) {
        Serial.println("[WIFI] Failed to connect, restarting...");
        tft.fillScreen(ST77XX_BLACK);
        tft.setCursor(0, 40);
        tft.setTextColor(ST77XX_RED);
        tft.setTextSize(1);
        tft.println("WiFi FAILED!");
        tft.println("Restarting...");
        delay(3000);
        ESP.restart();
    }

    Serial.printf("[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

    tft.fillScreen(ST77XX_BLACK);
    tft.setCursor(0, 10);
    tft.setTextColor(ST77XX_GREEN);
    tft.setTextSize(1);
    tft.println("WiFi: OK");
    tft.setTextColor(ST77XX_WHITE);
    tft.print("IP: ");
    tft.println(WiFi.localIP().toString().c_str());
    delay(1500);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    // Debug: In ra topic nhận được
    Serial.printf("[MQTT] Received topic: %s\n", topic);
    Serial.printf("[MQTT] Payload length: %d\n", length);
    
    // Parse JSON payload
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, payload, length);

    if (error) {
        Serial.printf("[MQTT] Failed to parse JSON: %s\n", error.c_str());
        // Try to print raw payload
        Serial.print("[MQTT] Raw payload: ");
        for(unsigned int i = 0; i < length && i < 100; i++) {
            Serial.print((char)payload[i]);
        }
        Serial.println();
        return;
    }

    String topicStr = String(topic);
    String vehicleInfoTopic = String("parking/payment/") + GATE_ID + "/vehicle_info";
    String addCardModeTopic = String("parking/payment/") + GATE_ID + "/add_card_mode";

    // Nhận thông tin xe từ Python app - GIỐNG WEB
    if (topicStr == vehicleInfoTopic) {
        // QUAN TRỌNG: Reset toàn bộ trước khi nhận dữ liệu mới
        currentVehicle = {"", "", "", 0, false, false, ""};
        
        // Nhận dữ liệu mới từ server
        currentVehicle.plate = doc["plate"].as<String>();
        currentVehicle.timeIn = doc["time_in"].as<String>();
        currentVehicle.timeOut = doc["time_out"].as<String>();
        currentVehicle.fee = doc["fee"].as<int>();  // calc_fee từ server (runtime calculation)
        currentVehicle.isPaid = doc["paid"].as<bool>();
        currentVehicle.hasData = true;

        Serial.println("[MQTT] ===== VEHICLE INFO RECEIVED =====");
        Serial.printf("  Plate: %s\n", currentVehicle.plate.c_str());
        Serial.printf("  Time In: %s\n", currentVehicle.timeIn.c_str());
        Serial.printf("  Time Out: %s\n", currentVehicle.timeOut.c_str());
        Serial.printf("  Fee (calc_fee): %d VND\n", currentVehicle.fee);
        Serial.printf("  Already paid: %s\n", currentVehicle.isPaid ? "YES" : "NO");
        
        // Tự động chuyển sang hiển thị thông tin nếu đang chờ
        if (currentState == STATE_WAITING && currentVehicle.hasData) {
            if (currentVehicle.isPaid) {
                displayAlreadyPaid();
                currentState = STATE_ALREADY_PAID;
            } else {
                displayVehicleInfo();
                currentState = STATE_SHOW_INFO;
            }
            stateTimeout = millis();
        }
    }
    // Nhận lệnh bật/tắt chế độ thêm thẻ
    else if (topicStr == addCardModeTopic) {
        bool enabled = doc["enabled"].as<bool>();
        addCardModeEnabled = enabled;

        Serial.printf("[MQTT] Add card mode: %s\n", enabled ? "ENABLED" : "DISABLED");

        if (enabled) {
            // Hiển thị màn hình chờ quét thẻ - sẽ gọi sau
            Serial.println("[MQTT] Add card mode ENABLED - will display add card screen");
            addCardModeStartTime = millis();  // Ghi nhận thời gian bắt đầu
        } else {
            // Quay lại màn hình chờ bình thường - sẽ gọi sau
            Serial.println("[MQTT] Add card mode DISABLED - will display welcome screen");
            currentState = STATE_WAITING;
            addCardModeStartTime = 0;  // Reset thời gian
        }
    }
    // Nhận QR data từ server - ĐỒNG BỘ VỚI WEB
    else if (topicStr == (String("parking/payment/") + GATE_ID + "/qr_data")) {
        // QR data is now handled by desktop application (parking_ui.py)
        // ESP32 no longer stores or displays QR codes, so we ignore this data
        Serial.println("[MQTT] QR data received but ignored (ESP32 no longer displays QR)");
    }
    // Nhận thông báo thanh toán thành công từ SePay webhook
    else if (topicStr == (String("parking/payment/") + GATE_ID + "/payment_confirmed")) {
        Serial.println("[MQTT] ===== PAYMENT CONFIRMED =====");
        int session_id = doc["session_id"].as<int>();
        String status = doc["status"].as<String>();
        int amount = doc["amount"].as<int>();

        Serial.printf("  Session: #%d\n", session_id);
        Serial.printf("  Status: %s\n", status.c_str());
        Serial.printf("  Amount: %d VND\n", amount);

        // Kiểm tra thanh toán thành công và hiện tại đang ở trạng thái chờ thanh toán
        if (status == "paid" && currentState == STATE_SHOW_QR && currentVehicle.hasData) {
            Serial.println("[SUCCESS] Payment confirmed by SePay!");
            // Chuyển sang màn hình đã thanh toán
            displayPaymentSuccess();
            currentState = STATE_SHOW_PAID;
            stateTimeout = millis();
            currentVehicle.isPaid = true;
        }
    }
    else {
        Serial.printf("[MQTT] Unknown topic: %s\n", topic);
    }
}

void mqttConnect() {
    // ✅ Add retry limit to prevent infinite blocking
    int retries = 0;
    const int MAX_RETRIES = 3;

    while (!mqtt.connected() && retries < MAX_RETRIES) {
        Serial.print("[MQTT] Connecting...");
        String clientId = "esp32-payment-" + String(random(0xffff), HEX);

        if (mqtt.connect(clientId.c_str())) {
            Serial.println("connected!");

            // Subscribe to vehicle info topic
            String vehicleTopic = String("parking/payment/") + GATE_ID + "/vehicle_info";
            mqtt.subscribe(vehicleTopic.c_str(), 1);
            Serial.printf("[MQTT] Subscribed to: %s\n", vehicleTopic.c_str());

            // Subscribe to add card mode topic
            String addCardTopic = String("parking/payment/") + GATE_ID + "/add_card_mode";
            mqtt.subscribe(addCardTopic.c_str(), 1);
            Serial.printf("[MQTT] Subscribed to: %s\n", addCardTopic.c_str());

            // Subscribe to QR data topic
            String qrDataTopic = String("parking/payment/") + GATE_ID + "/qr_data";
            mqtt.subscribe(qrDataTopic.c_str(), 1);
            Serial.printf("[MQTT] Subscribed to: %s\n", qrDataTopic.c_str());

            // Subscribe to payment confirmed topic (from SePay webhook)
            String paymentTopic = String("parking/payment/") + GATE_ID + "/payment_confirmed";
            mqtt.subscribe(paymentTopic.c_str(), 1);
            Serial.printf("[MQTT] Subscribed to: %s\n", paymentTopic.c_str());

            // Publish online status
            JsonDocument doc;
            doc["online"] = true;
            doc["mac"] = WiFi.macAddress();
            doc["ip"] = WiFi.localIP().toString();
            String json;
            serializeJson(doc, json);

            String topic = String("parking/payment/") + GATE_ID + "/status";
            mqtt.publish(topic.c_str(), json.c_str(), true);

            Serial.printf("[MQTT] Published to: %s\n", topic.c_str());
        } else {
            Serial.printf("failed, rc=%d\n", mqtt.state());
            retries++;
            if (retries < MAX_RETRIES) {
                ArduinoOTA.handle();  // ✅ Handle OTA even during MQTT reconnect
                delay(2000);
            }
        }
    }

    if (!mqtt.connected()) {
        Serial.println("[MQTT] ❌ Failed after max retries");
    }
}

void sendHeartbeat() {
    JsonDocument doc;
    doc["time"] = millis();
    doc["mac"] = WiFi.macAddress();
    doc["ip"] = WiFi.localIP().toString();
    String json;
    serializeJson(doc, json);

    String topic = String("parking/payment/") + GATE_ID + "/heartbeat";
    mqtt.publish(topic.c_str(), json.c_str());
}

void sendCardScanned(String cardUID) {
    JsonDocument doc;
    doc["mac"] = WiFi.macAddress();
    doc["ip"] = WiFi.localIP().toString();
    doc["card_id"] = cardUID;
    doc["time"] = millis();
    String json;
    serializeJson(doc, json);

    String topic = String("parking/payment/") + GATE_ID + "/card_scanned";
    mqtt.publish(topic.c_str(), json.c_str());

    Serial.printf("[MQTT] Card scanned: %s\n", cardUID.c_str());
}

// ====== TFT DISPLAY FUNCTIONS ======
// Reset tất cả dữ liệu
void resetAllData() {
    // Reset về giá trị mặc định
    currentVehicle = {"", "", "", 0, false, false, ""};
    // QR data no longer stored on ESP32 (now handled by desktop app)
    Serial.println("[RESET] All data cleared!");
}

void displayWelcomeScreen() {
    tft.fillScreen(ST77XX_BLACK);

    // Viền trên
    tft.drawRect(0, 0, 128, 160, ST77XX_CYAN);
    tft.drawRect(1, 1, 126, 158, ST77XX_CYAN);

    // Tiêu đề
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_YELLOW);
    tft.setCursor(18, 10);
    tft.println("BAI DO XE ABC");

    // Đường phân cách
    tft.drawLine(5, 28, 123, 28, ST77XX_CYAN);

    // Nội dung chính
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(1);

    tft.setCursor(20, 50);
    tft.println("Quet the de");

    tft.setCursor(20, 65);
    tft.println("xem thong tin");

    // Icon thẻ (vẽ đơn giản)
    tft.drawRect(45, 90, 38, 25, ST77XX_GREEN);
    tft.fillRect(47, 92, 34, 21, ST77XX_BLACK);
    tft.drawLine(50, 95, 76, 95, ST77XX_GREEN);
    tft.drawLine(50, 100, 70, 100, ST77XX_GREEN);
    tft.drawLine(50, 105, 65, 105, ST77XX_GREEN);

    // Footer
    tft.setTextColor(ST77XX_CYAN);
    tft.setTextSize(1);
    tft.setCursor(28, 145);
    tft.println("RFID Ready");
}

void displayAddCardMode() {
    tft.fillScreen(ST77XX_BLACK);

    // Viền xanh dương
    tft.drawRect(0, 0, 128, 160, ST77XX_BLUE);
    tft.drawRect(1, 1, 126, 158, ST77XX_BLUE);

    // Tiêu đề
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_BLUE);
    tft.setCursor(12, 10);
    tft.println("CHE DO THEM THE");

    // Đường phân cách
    tft.drawLine(5, 28, 123, 28, ST77XX_BLUE);

    // Animation loading
    int centerX = 64;
    int centerY = 60;
    
    // Vẽ vòng tròn xoay
    for (int i = 0; i < 8; i++) {
        float angle = (i * 45) * PI / 180.0;
        int x = centerX + 20 * cos(angle);
        int y = centerY + 20 * sin(angle);
        
        uint16_t color = (i < 3) ? ST77XX_BLUE : 0x0208; // Xanh nhạt
        tft.fillCircle(x, y, 2, color);
    }

    // Nội dung chính
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(1);

    tft.setCursor(15, 100);
    tft.println("Dang cho quet");

    tft.setCursor(25, 115);
    tft.println("the moi...");

    // Footer
    tft.setTextColor(ST77XX_BLUE);
    tft.setTextSize(1);
    tft.setCursor(20, 145);
    tft.println("Web Interface");
}

// Hàm riêng để update countdown và animation (không làm nháy màn hình)
void updateAddCardCountdown() {
    static unsigned long lastAnimationUpdate = 0;
    static int animationFrame = 0;
    
    // Update animation loading mỗi 200ms
    if (millis() - lastAnimationUpdate > 200) {
        // Xóa vùng animation cũ
        tft.fillCircle(64, 60, 25, ST77XX_BLACK);
        
        // Vẽ animation mới
        int centerX = 64;
        int centerY = 60;
        for (int i = 0; i < 8; i++) {
            float angle = ((i + animationFrame) * 45) * PI / 180.0;
            int x = centerX + 20 * cos(angle);
            int y = centerY + 20 * sin(angle);
            
            uint16_t color = (i < 3) ? ST77XX_BLUE : 0x0208; // Xanh nhạt
            tft.fillCircle(x, y, 2, color);
        }
        
        animationFrame = (animationFrame + 1) % 8;
        lastAnimationUpdate = millis();
    }
    
    // Update countdown chỉ khi giây thay đổi
    static int lastRemainingSeconds = -1;
    if (addCardModeStartTime > 0) {
        unsigned long elapsed = millis() - addCardModeStartTime;
        unsigned long remaining = (elapsed < ADD_CARD_TIMEOUT) ? 
                                 (ADD_CARD_TIMEOUT - elapsed) / 1000 : 0;
        
        // Chỉ update khi giây thay đổi
        if ((int)remaining != lastRemainingSeconds) {
            lastRemainingSeconds = (int)remaining;
            
            // Xóa vùng countdown cũ
            tft.fillRect(30, 125, 70, 15, ST77XX_BLACK);
            
            // Vẽ countdown mới
            tft.setTextColor(ST77XX_YELLOW);
            tft.setTextSize(1);
            tft.setCursor(35, 130);
            tft.printf("Timeout: %ds", (int)remaining);
        }
    }
}

void displayVehicleInfo(bool fullRedraw) {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_CYAN);
    tft.drawRect(1, 1, 126, 158, ST77XX_CYAN);
    
    // Tiêu đề
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_CYAN);
    tft.setCursor(20, 8);
    tft.println("THONG TIN XE");
    
    // Ngăn cách
    tft.drawLine(5, 25, 123, 25, ST77XX_CYAN);
    
    // Biển số
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(5, 35);
    tft.print("Bien so: ");
    tft.setTextColor(ST77XX_YELLOW);
    tft.println(currentVehicle.plate);
    
    // Thời gian vào
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(5, 50);
    tft.print("Gio vao: ");
    tft.setTextColor(ST77XX_GREEN);
    tft.println(currentVehicle.timeIn);
    
    // Thời gian ra
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(5, 65);
    tft.print("Gio ra: ");
    tft.setTextColor(ST77XX_GREEN);
    tft.println(currentVehicle.timeOut);
    
    // Ngăn cách
    tft.drawLine(5, 82, 123, 82, ST77XX_CYAN);
    
    // Số tiền
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_YELLOW);
    
    // Debug trước khi hiển thị số tiền
    Serial.printf("[DISPLAY VEHICLE] Fee from DB: %d VND\n", currentVehicle.fee);
    
    // Format số tiền với dấu phẩy
    char moneyBuffer[30];
    formatMoney(moneyBuffer, currentVehicle.fee);
    
    // Tính toán để căn giữa
    int moneyLen = strlen(moneyBuffer) + 4; // +4 cho " VND"
    int cursorX = (128 - moneyLen * 12) / 2; // 12 = width of size-2 char
    if (cursorX < 5) cursorX = 5;
    
    tft.setCursor(cursorX, 90);
    tft.print(moneyBuffer);
    tft.print(" VND");
    
    // Hướng dẫn nhấn nút
    tft.fillRect(5, 120, 118, 35, ST77XX_GREEN);
    tft.drawRect(5, 120, 118, 35, ST77XX_WHITE);
    
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_BLACK);
    tft.setCursor(10, 130);
    tft.println("NHAN NUT DE");
    tft.setCursor(10, 142);
    tft.println("XEM MA QR");
}

void displayPaymentProcessing() {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_CYAN);
    tft.drawRect(1, 1, 126, 158, ST77XX_CYAN);

    // Tiêu đề
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_CYAN);
    tft.setCursor(12, 10);
    tft.println("DANG XU LY...");

    // Animation vòng tròn loading
    int centerX = 64;
    int centerY = 60;
    int radius = 25;

    for (int i = 0; i < 8; i++) {
        float angle = (i * 45) * PI / 180.0;
        int x = centerX + radius * cos(angle);
        int y = centerY + radius * sin(angle);

        // Vẽ các chấm với độ mờ khác nhau
        uint16_t color;
        if (i < 2) color = ST77XX_CYAN;
        else if (i < 4) color = 0x0410; // Xanh nhạt
        else if (i < 6) color = 0x0208; // Xanh rất nhạt
        else color = 0x0104; // Xanh cực nhạt

        tft.fillCircle(x, y, 3, color);
        delay(80);
    }

    // Hiển thị biển số
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_YELLOW);
    tft.setCursor(5, 100);
    tft.println("Bien so:");
    tft.setTextSize(2);
    tft.setCursor(15, 115);
    tft.println(currentVehicle.plate);

    delay(500);
}

void displayAlreadyPaid() {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_ORANGE);
    tft.drawRect(1, 1, 126, 158, ST77XX_ORANGE);
    
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_ORANGE);
    tft.setCursor(15, 50);
    tft.println("DA THANH");
    tft.setCursor(30, 70);
    tft.println("TOAN!");
    
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(15, 100);
    tft.println("Bien: ");
    tft.setTextColor(ST77XX_YELLOW);
    tft.println(currentVehicle.plate);
    
    tft.setTextColor(ST77XX_GREEN);
    tft.setCursor(15, 120);
    tft.println("Cam on ban!");
}

void displayPaymentSuccess() {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_GREEN);
    tft.drawRect(1, 1, 126, 158, ST77XX_GREEN);
    
    // Vòng tròn xanh + checkmark
    int centerX = 64;
    int centerY = 55;
    tft.fillCircle(centerX, centerY, 30, ST77XX_GREEN);
    
    // Dấu tick
    tft.drawLine(52, 55, 58, 61, ST77XX_BLACK);
    tft.drawLine(53, 55, 59, 61, ST77XX_BLACK);
    tft.drawLine(54, 55, 60, 61, ST77XX_BLACK);
    tft.drawLine(58, 61, 76, 43, ST77XX_BLACK);
    tft.drawLine(59, 61, 77, 43, ST77XX_BLACK);
    tft.drawLine(60, 61, 78, 43, ST77XX_BLACK);
    
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_GREEN);
    tft.setCursor(8, 100);
    tft.println("THANH CONG");
    
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(15, 130);
    tft.printf("%d VND", currentVehicle.fee);
    
    tft.setCursor(15, 145);
    tft.setTextColor(ST77XX_CYAN);
    tft.println("Cam on ban!");
}

void displayPaymentError() {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_RED);
    tft.drawRect(1, 1, 126, 158, ST77XX_RED);

    int centerX = 64;
    int centerY = 55;
    int radius = 30;

    // Animation: Vẽ vòng tròn đỏ từng đoạn
    for (int angle = 0; angle <= 360; angle += 10) {
        float rad = angle * PI / 180.0;
        int x = centerX + radius * cos(rad - PI/2);
        int y = centerY + radius * sin(rad - PI/2);
        tft.fillCircle(x, y, 2, ST77XX_RED);
        delay(5);
    }

    // Điền vòng tròn
    tft.fillCircle(centerX, centerY, radius, ST77XX_RED);
    delay(200);

    // Animation: Vẽ dấu X từng nét (cân đối)
    int xSize = 20;  // Kích thước X
    int xLeft = centerX - xSize/2;
    int xRight = centerX + xSize/2;
    int yTop = centerY - xSize/2;
    int yBottom = centerY + xSize/2;

    // Nét 1: từ trái trên xuống phải dưới (\)
    for (int i = 0; i <= xSize; i++) {
        int x = xLeft + i;
        int y = yTop + i;
        tft.drawLine(xLeft, yTop, x, y, ST77XX_BLACK);
        tft.drawLine(xLeft+1, yTop, x+1, y, ST77XX_BLACK);
        tft.drawLine(xLeft+2, yTop, x+2, y, ST77XX_BLACK);
        delay(15);
    }

    // Nét 2: từ phải trên xuống trái dưới (/)
    for (int i = 0; i <= xSize; i++) {
        int x = xRight - i;
        int y = yTop + i;
        tft.drawLine(xRight, yTop, x, y, ST77XX_BLACK);
        tft.drawLine(xRight-1, yTop, x-1, y, ST77XX_BLACK);
        tft.drawLine(xRight-2, yTop, x-2, y, ST77XX_BLACK);
        delay(15);
    }

    delay(300);

    // Text "KHÔNG THÀNH CÔNG"
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(12, 100);
    tft.println("KHONG THANH");
    delay(100);
    tft.setCursor(36, 112);
    tft.println("CONG!");

    // Lý do
    delay(200);
    tft.setTextSize(1);
    tft.fillCircle(centerX, centerY, 25, ST77XX_GREEN);
    tft.drawCircle(centerX, centerY, 25, ST77XX_WHITE);

    // Vẽ dấu tick đơn giản
    tft.drawLine(52, 50, 58, 56, ST77XX_BLACK);
    tft.drawLine(53, 50, 59, 56, ST77XX_BLACK);
    tft.drawLine(54, 50, 60, 56, ST77XX_BLACK);
    
    tft.drawLine(58, 56, 76, 38, ST77XX_BLACK);
    tft.drawLine(58, 57, 76, 39, ST77XX_BLACK);
    tft.drawLine(58, 58, 76, 40, ST77XX_BLACK);

    // Text chính
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_GREEN);
    tft.setCursor(15, 85);
    tft.println("THANH TOAN");
    
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(25, 100);
    tft.println("THANH CONG!");

    // Biển số
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_YELLOW);
    tft.setCursor(8, 120);
    tft.print("Bien so: ");
    tft.setTextColor(ST77XX_WHITE);
    tft.println(currentVehicle.plate);

    // Hướng dẫn
    tft.setTextColor(ST77XX_CYAN);
    tft.setCursor(20, 140);
    tft.println("Xin cam on!");
}

void displayLoadingScreen() {
    tft.fillScreen(ST77XX_BLACK);
    tft.drawRect(0, 0, 128, 160, ST77XX_CYAN);
    
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_CYAN);
    tft.setCursor(20, 50);
    tft.println("Dang tai...");
    
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(10, 70);
    tft.println("Vui long cho");
    
    // Animation dots
    static int dots = 0;
    tft.setCursor(10, 85);
    for(int i = 0; i < dots % 4; i++) {
        tft.print(".");
    }
    dots++;
}


// ====== RFID SETUP ======
void setup_rfid_rc522(){
    // Khởi tạo MFRC522 với Arduino library (đơn giản hơn)
    mfrc522.PCD_Init();
    
    // Kiểm tra version
    byte version = mfrc522.PCD_ReadRegister(mfrc522.VersionReg);
    Serial.printf("[RFID] RC522 Version: 0x%02X\n", version);
    
    if (version == 0x00 || version == 0xFF) {
        Serial.println("[ERROR] RC522 not found! Check wiring.");
    } else {
        Serial.println("[RFID] RC522 initialized successfully");
        
        // Test self-test
        bool selfTestResult = mfrc522.PCD_PerformSelfTest();
        Serial.printf("[RFID] Self-test: %s\n", selfTestResult ? "PASS" : "FAIL");
        
        // Re-init sau self-test
        mfrc522.PCD_Init();
    }
}

// ====== RFID READ FUNCTION (cho main loop) ======
String readRFIDCard() {
    static String lastUID = "";
    static unsigned long lastReadTime = 0;
    const unsigned long DEBOUNCE_TIME = 3000; // 3 giây debounce
    
    // Kiểm tra có thẻ mới không
    if (!mfrc522.PICC_IsNewCardPresent()) {
        return "";
    }
    
    // Đọc thẻ
    if (!mfrc522.PICC_ReadCardSerial()) {
        return "";
    }
    
    // Tạo UID string
    String uidString = "";
    for (byte i = 0; i < mfrc522.uid.size; i++) {
        if (i > 0) uidString += "-";
        if (mfrc522.uid.uidByte[i] < 0x10) uidString += "0";
        uidString += String(mfrc522.uid.uidByte[i], HEX);
    }
    uidString.toUpperCase();
    
    // Halt thẻ
    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
    
    // Debounce: Kiểm tra thẻ trùng trong 3 giây
    unsigned long currentTime = millis();
    if (uidString == lastUID && (currentTime - lastReadTime) < DEBOUNCE_TIME) {
        return ""; // Bỏ qua thẻ trùng
    }
    
    // Cập nhật thông tin thẻ cuối
    lastUID = uidString;
    lastReadTime = currentTime;
    
    Serial.printf("[RFID] Card detected: %s\n", uidString.c_str());
    return uidString;
}

// ====== SETUP ======
void setup() {
    // Init Serial
    Serial.begin(115200);
    Serial.println("\n\n=== PAYMENT TERMINAL ===\n");
    
    // Reset tất cả dữ liệu khi bắt đầu
    resetAllData();
    Serial.println("[BOOT] All data initialized.");
    
    Serial.println("\n[SYSTEM] Payment Terminal Starting...");

    // Khởi tạo SPI với pins đúng cho cả TFT và RFID (dùng chung bus)
    SPI.begin(RC522_SCK, RC522_MISO, RC522_MOSI, -1);
    Serial.println("[SPI] Arduino SPI initialized with custom pins");
    
    // Khởi tạo RFID với Arduino MFRC522 library
    Serial.println("[RFID] Initializing RC522...");
    setup_rfid_rc522();

    Serial.println("[TFT] Initializing ST7735...");

    tft.initR(INITR_BLACKTAB);

    tft.setRotation(00);  // ✅ Rotation 2 = 180 degrees flip
    tft.fillScreen(ST77XX_BLACK);

    // Test màu để debug
    delay(100);
    tft.fillScreen(ST77XX_RED);
    delay(500);
    tft.fillScreen(ST77XX_GREEN);
    delay(500);
    tft.fillScreen(ST77XX_BLUE);
    delay(500);
    tft.fillScreen(ST77XX_BLACK);

    tft.setTextColor(ST77XX_GREEN);
    tft.setTextSize(1);
    tft.setCursor(0, 10);
    tft.println("Payment Terminal");
    tft.println("Khoi dong...");

    Serial.println("[TFT] ST7735 initialized");

    // RFID sẽ được đọc trong main loop (không cần RTOS task)
    Serial.println("[RFID] Ready for card reading");

    // Kết nối WiFi
    wifiConnect();

    // Setup OTA (sau khi WiFi đã kết nối)
    setupOTA();

    // Kết nối MQTT
    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    mqtt.setCallback(mqttCallback);
    mqttConnect();

    // Hiển thị màn hình chờ
    displayWelcomeScreen();
    currentState = STATE_WAITING;

    Serial.println("[SYSTEM] All systems ready!");
    Serial.println("[SYSTEM] Waiting for RFID card...");
}

// ====== LOOP ======
void loop() {
    // Xử lý OTA (quan trọng: phải gọi thường xuyên)
    ArduinoOTA.handle();

    // Kiểm tra WiFi
    if (WiFi.status() != WL_CONNECTED) {
        wifiConnect();
    }

    // Kiểm tra MQTT
    if (!mqtt.connected()) {
        mqttConnect();
    }
    mqtt.loop();

    // Gửi heartbeat
    if (millis() - lastHeartbeat > HEARTBEAT_INTERVAL) {
        sendHeartbeat();
        lastHeartbeat = millis();
    }

    // Nếu đang OTA update, không chạy state machine (để không vẽ đè lên màn hình OTA)
    if (isOTAUpdating) {
        delay(10);  // Nhỏ delay để không chiếm CPU
        return;
    }

    // Kiểm tra và hiển thị màn hình phù hợp với chế độ
    static bool lastAddCardMode = false;
    static unsigned long lastScreenUpdate = 0;
    
    if (addCardModeEnabled != lastAddCardMode) {
        lastAddCardMode = addCardModeEnabled;
        if (addCardModeEnabled) {
            Serial.println("[LOOP] Switching to ADD CARD mode - displaying add card screen");
            displayAddCardMode();
            // Hiển thị countdown lần đầu
            updateAddCardCountdown();
        } else {
            Serial.println("[LOOP] Switching to NORMAL mode - displaying welcome screen");
            displayWelcomeScreen();
            currentState = STATE_WAITING;
        }
        lastScreenUpdate = millis();
    }
    
    // Kiểm tra timeout chế độ thêm thẻ (30 giây)
    if (addCardModeEnabled && addCardModeStartTime > 0 && 
        (millis() - addCardModeStartTime > ADD_CARD_TIMEOUT)) {
        Serial.println("[TIMEOUT] Add card mode timeout - switching to normal mode");
        addCardModeEnabled = false;
        addCardModeStartTime = 0;
        // Logic chuyển đổi sẽ được xử lý ở phần kiểm tra addCardModeEnabled != lastAddCardMode
    }
    
    // Update countdown và animation (mượt mà, không nháy màn hình)
    if (addCardModeEnabled && (millis() - lastScreenUpdate > 200)) {
        updateAddCardCountdown();
        lastScreenUpdate = millis();
    }

    // State Machine
    switch (currentState) {
        case STATE_WAITING: {
            // Đọc thẻ RFID
            String cardUID = readRFIDCard();

            if (cardUID != "" && cardUID != lastCardUID) {
                Serial.printf("[RFID] Card detected: %s\n", cardUID.c_str());

                // Kiểm tra chế độ thêm thẻ
                if (addCardModeEnabled) {
                    // Chế độ thêm thẻ: Gửi trực tiếp card UID
                    Serial.println("[ADD_CARD] Sending card UID for adding...");
                    sendCardScanned(cardUID);
                    
                    // Hiển thị thông báo đã gửi
                    tft.fillScreen(ST77XX_BLACK);
                    tft.drawRect(0, 0, 128, 160, ST77XX_GREEN);
                    tft.setTextSize(1);
                    tft.setTextColor(ST77XX_GREEN);
                    tft.setCursor(15, 60);
                    tft.println("DA GUI THE!");
                    tft.setTextColor(ST77XX_WHITE);
                    tft.setCursor(10, 80);
                    tft.println(cardUID);
                    delay(2000);
                    
                    // Tự động thoát chế độ thêm thẻ sau khi gửi thành công
                    Serial.println("[CARD_SENT] Auto-disabling add card mode after successful send");
                    addCardModeEnabled = false;
                    addCardModeStartTime = 0;
                } else {
                    // Chế độ bình thường: Gửi card_id cho server (GIỐNG WEB)
                    // Server sẽ:
                    // 1. Query database lấy session_id, calc_fee
                    // 2. Gửi vehicle_info
                    // 3. Tạo QR từ SePay và gửi qr_data
                    
                    // Reset toàn bộ dữ liệu cũ
                    resetAllData();
                    
                    // Lưu card UID
                    currentVehicle.cardUID = cardUID;
                    
                    if (mqtt.connected()) {
                        sendCardScanned(cardUID);
                        Serial.println("[CARD] Sent to server, waiting for vehicle_info and qr_data...");
                    }
                }

                // Chỉ đợi thông tin xe khi KHÔNG ở chế độ thêm thẻ
                if (!addCardModeEnabled) {
                    // Hiển thị màn hình loading
                    displayLoadingScreen();
                    
                    // Đợi vehicle_info từ server
                    // QR data is no longer processed on ESP32 (handled by desktop app)
                    Serial.println("[WAIT] Waiting for vehicle_info from server...");
                    unsigned long waitStart = millis();
                    unsigned long lastDotUpdate = 0;

                    while (millis() - waitStart < 5000) {  // Đợi 5 giây cho vehicle_info
                        // ✅ Critical: Process OTA, MQTT, Heartbeat during wait
                        ArduinoOTA.handle();
                        mqtt.loop();
                        sendHeartbeat();

                        // Update loading animation mỗi 500ms
                        if (millis() - lastDotUpdate > 500) {
                            displayLoadingScreen();
                            lastDotUpdate = millis();
                        }

                        // Kiểm tra vehicle_info đã có
                        if (currentVehicle.hasData) {
                            Serial.println("[WAIT] Received vehicle_info!");
                            break;
                        }
                        delay(10);  // ✅ Reduced from 50ms to 10ms for better responsiveness
                    }

                    // Debug: In ra trạng thái nhận được
                    Serial.printf("[DEBUG] Vehicle data received: %s\n", 
                                  currentVehicle.hasData ? "YES" : "NO");

                    if (currentVehicle.hasData) {
                        // Kiểm tra đã thanh toán chưa
                        if (currentVehicle.isPaid) {
                            // ĐÃ THANH TOÁN RỒI → Hiển thị màn hình đặc biệt
                            Serial.println("[STATE] -> ALREADY_PAID");
                            displayAlreadyPaid();
                            currentState = STATE_ALREADY_PAID;
                            stateTimeout = millis();
                        } else {
                            // CHƯA THANH TOÁN → Hiển thị thông tin xe
                            Serial.println("[STATE] -> SHOW_INFO");
                            displayVehicleInfo();
                            currentState = STATE_SHOW_INFO;
                            stateTimeout = millis();
                        }
                    } else {
                        // Không nhận được thông tin xe
                        Serial.println("[ERROR] No vehicle info received");
                        
                        // Hiển thị lỗi
                        tft.fillScreen(ST77XX_BLACK);
                        tft.drawRect(0, 0, 128, 160, ST77XX_RED);
                        
                        tft.setTextSize(1);
                        tft.setTextColor(ST77XX_RED);
                        tft.setCursor(18, 50);
                        tft.println("Khong co");
                        tft.setCursor(18, 65);
                        tft.println("thong tin xe");
                        
                        // Hiển thị debug info
                        tft.setTextSize(1);
                        tft.setTextColor(ST77XX_YELLOW);
                        tft.setCursor(5, 90);
                        tft.printf("Card: %s", cardUID.c_str());
                        
                        tft.setTextColor(ST77XX_WHITE);
                        tft.setCursor(5, 105);
                        tft.println("Kiem tra:");
                        tft.setCursor(5, 120);
                        tft.println("khong co thong tin");

                        delay(3000);
                        displayWelcomeScreen();
                    }
                } // Kết thúc if (!addCardModeEnabled)

                lastCardUID = cardUID;
                lastCardTime = millis();
            } else if (cardUID == "" && (millis() - lastCardTime > CARD_DEBOUNCE)) {
                lastCardUID = "";
            }
            break;
        }

        case STATE_SHOW_INFO: {
            // Auto-transition to STATE_SHOW_QR after 3 seconds
            if (millis() - stateTimeout > 3000) {
                Serial.println("[STATE] Auto transition INFO -> SHOW_QR (desktop app will show QR)");

                // Hiển thị thông báo: Vui lòng quét QR trên app desktop
                tft.fillScreen(ST77XX_BLACK);
                tft.drawRect(0, 0, 128, 160, ST77XX_GREEN);
                tft.drawRect(1, 1, 126, 158, ST77XX_GREEN);

                tft.setTextSize(1);
                tft.setTextColor(ST77XX_GREEN);
                tft.setCursor(8, 20);
                tft.println("DANG CHO");
                tft.setCursor(15, 32);
                tft.println("THANH TOAN");

                tft.setTextColor(ST77XX_WHITE);
                tft.setCursor(5, 60);
                tft.println("Vui long quet QR");
                tft.setCursor(5, 75);
                tft.println("tren app desktop");

                tft.setTextColor(ST77XX_YELLOW);
                tft.setTextSize(1);
                tft.setCursor(10, 100);
                tft.println("Tien:");
                tft.setTextColor(ST77XX_CYAN);
                tft.setCursor(10, 115);
                char moneyBuffer[30];
                sprintf(moneyBuffer, "%d VND", currentVehicle.fee);
                tft.println(moneyBuffer);

                currentState = STATE_SHOW_QR;
                stateTimeout = millis();
                break;
            }

            // Timeout 30 giây - quay lại màn hình chờ
            if (millis() - stateTimeout > 30000) {
                Serial.println("[STATE] Info timeout -> WAITING");
                // Reset toàn bộ dữ liệu trước khi về màn hình chờ
                resetAllData();
                displayWelcomeScreen();
                currentState = STATE_WAITING;
            }
            break;
        }
        
// STATE_WAITING_QR removed - không cần nữa vì QR có sẵn từ lúc quét thẻ

        case STATE_SHOW_QR: {
            // Poll MQTT để kiểm tra thanh toán thành công
            static unsigned long lastPollTime = 0;
            if (millis() - lastPollTime > 1000) {  // Poll mỗi 1 giây
                mqtt.loop();
                lastPollTime = millis();
                
                // Kiểm tra nếu đã thanh toán (server sẽ gửi update)
                if (currentVehicle.isPaid) {
                    Serial.println("[PAYMENT] Success detected!");
                    displayPaymentSuccess();
                    currentState = STATE_SHOW_PAID;
                    stateTimeout = millis();
                    break;
                }
            }

            // Timeout 120 giây - Tự động hủy
            if (millis() - stateTimeout > 120000) {
                Serial.println("[STATE] QR timeout -> WAITING");
                currentVehicle.hasData = false;
                displayWelcomeScreen();
                currentState = STATE_WAITING;
            }
            break;
        }

        case STATE_PROCESSING: {
            // Kiểm tra có dữ liệu xe không
            if (!currentVehicle.hasData || currentVehicle.plate == "" || currentVehicle.plate == "KHONG TIM THAY") {
                // KHÔNG CÓ DỮ LIỆU → Hiển thị lỗi
                Serial.println("[STATE] Processing -> SHOW_ERROR (no data)");
                displayPaymentError();
                currentState = STATE_SHOW_ERROR;
                stateTimeout = millis();
            } else {
                // CÓ DỮ LIỆU → Thanh toán thành công
                Serial.println("[STATE] Processing -> SHOW_PAID");
                displayPaymentSuccess();
                currentState = STATE_SHOW_PAID;
                stateTimeout = millis();

                // Gửi thông báo thanh toán thành công qua MQTT
                if (mqtt.connected()) {
                    JsonDocument doc;
                    doc["mac"] = WiFi.macAddress();
                    doc["card_id"] = lastCardUID;
                    doc["plate"] = currentVehicle.plate;
                    doc["paid"] = true;
                    String json;
                    serializeJson(doc, json);

                    String topic = String("parking/payment/") + GATE_ID + "/payment_confirmed";
                    mqtt.publish(topic.c_str(), json.c_str());
                    Serial.println("[MQTT] Payment confirmed sent");
                }
            }
            break;
        }

        case STATE_SHOW_ERROR: {
            // Hiển thị lỗi 3 giây rồi quay lại màn hình chờ
            if (millis() - stateTimeout > 3000) {
                Serial.println("[STATE] Error done -> WAITING");
                currentVehicle.hasData = false;
                displayWelcomeScreen();
                currentState = STATE_WAITING;
            }
            break;
        }

        case STATE_SHOW_PAID: {
            // Hiển thị 5 giây rồi quay lại màn hình chờ
            if (millis() - stateTimeout > 5000) {
                Serial.println("[STATE] Done -> WAITING");
                currentVehicle.hasData = false;
                displayWelcomeScreen();
                currentState = STATE_WAITING;
            }
            break;
        }

        case STATE_ALREADY_PAID: {
            // Hiển thị 3 giây rồi quay lại màn hình chờ
            if (millis() - stateTimeout > 3000) {
                Serial.println("[STATE] Already paid -> WAITING");
                currentVehicle.hasData = false;
                displayWelcomeScreen();
                currentState = STATE_WAITING;
            }
            break;
        }
    }

    delay(50);
}