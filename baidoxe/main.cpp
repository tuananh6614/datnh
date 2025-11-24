#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <MFRC522.h>
#include "ota.h"
#include <WiFiManager.h>  // ✅ WiFiManager for easy WiFi configuration

// ====== WIFI & MQTT CONFIG ======
// WiFi credentials now managed by WiFiManager (no hardcoded SSID/PASS)

const char* MQTT_HOST = "linuxtuananh.zapto.org";  // ✅ Updated to use domain name for remote access
const int   MQTT_PORT = 1883;
const char* GATE_ID   = "gate01";

WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long lastHeartbeat = 0;
const unsigned long HEARTBEAT_INTERVAL = 4000;

// ====== RFID CONFIG (Arduino MFRC522 style như thanhtoan) ======
#define RC522_MOSI  11
#define RC522_MISO  13
#define RC522_SCK   12

#define RC522_CS_A  14   // RFID A (IN)
#define RC522_RST_A 15

#define RC522_CS_B  10   // RFID B (OUT)  
#define RC522_RST_B 9

// MFRC522 instances (như thanhtoan)
MFRC522 mfrc522_A(RC522_CS_A, RC522_RST_A);
MFRC522 mfrc522_B(RC522_CS_B, RC522_RST_B);

// ====== RFID DEBOUNCE DATA (như thanhtoan) ======
struct RFIDReader {
    MFRC522* device;
    String name;
    String trigger_type;
    String lastUID;
    unsigned long lastReadTime;
    const unsigned long DEBOUNCE_TIME = 3000; // 3 giây
};

RFIDReader rfid_A;
RFIDReader rfid_B;

// ====== WIFI & MQTT FUNCTIONS ======
void wifiConnect() {
    Serial.println("[WIFI] Starting WiFiManager...");

    WiFiManager wm;

    // Timeout 180 giây, sau đó restart
    wm.setConfigPortalTimeout(180);

    // Tự động kết nối hoặc mở config portal
    // AP Name: "ESP32-BaiDoXe", Password: "12345678"
    bool connected = wm.autoConnect("ESP32-BaiDoXe", "12345678");

    if (!connected) {
        Serial.println("[WIFI] Failed to connect, restarting...");
        delay(3000);
        ESP.restart();
    }

    Serial.printf("[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
}

void mqttConnect() {
    while (!mqtt.connected()) {
        Serial.print("[MQTT] Connecting...");
        String clientId = "esp32-parking-" + String(random(0xffff), HEX);
        if (mqtt.connect(clientId.c_str())) {
            Serial.println("connected!");
            JsonDocument doc;
            doc["online"] = true;
            doc["mac"] = WiFi.macAddress();
            doc["ip"] = WiFi.localIP().toString();
            String json;
            serializeJson(doc, json);
            String topic = String("parking/gate/") + GATE_ID + "/status";
            mqtt.publish(topic.c_str(), json.c_str(), true);
        } else {
            Serial.printf("failed, rc=%d\n", mqtt.state());
            delay(2000);
        }
    }
}

void sendHeartbeat() {
    JsonDocument doc;
    doc["time"] = millis();
    doc["mac"] = WiFi.macAddress();
    doc["ip"] = WiFi.localIP().toString();
    String json;
    serializeJson(doc, json);
    String topic = String("parking/gate/") + GATE_ID + "/heartbeat";
    mqtt.publish(topic.c_str(), json.c_str());
}

void sendTriggerWithCard(const char* type, const char* card_id) {
    JsonDocument doc;
    doc["mac"] = WiFi.macAddress();
    doc["ip"] = WiFi.localIP().toString();
    doc["card_id"] = card_id;
    doc["time"] = millis();
    String json;
    serializeJson(doc, json);

    String topic = String("parking/gate/") + GATE_ID + "/" + type;
    mqtt.publish(topic.c_str(), json.c_str());
    Serial.printf("[MQTT] Trigger %s with card: %s\n", type, card_id);
}

// ====== ARDUINO RFID FUNCTIONS (giống hệt thanhtoan) ======
String readRFIDCard(RFIDReader& reader) {
    // Kiểm tra có thẻ mới không
    if (!reader.device->PICC_IsNewCardPresent()) {
        return "";
    }
    
    // Đọc thẻ
    if (!reader.device->PICC_ReadCardSerial()) {
        return "";
    }
    
    // Tạo UID string
    String uidString = "";
    for (byte i = 0; i < reader.device->uid.size; i++) {
        if (i > 0) uidString += "-";
        if (reader.device->uid.uidByte[i] < 0x10) uidString += "0";
        uidString += String(reader.device->uid.uidByte[i], HEX);
    }
    uidString.toUpperCase();
    
    // Halt thẻ
    reader.device->PICC_HaltA();
    reader.device->PCD_StopCrypto1();
    
    // Debounce: Kiểm tra thẻ trùng trong 3 giây
    unsigned long currentTime = millis();
    if (uidString == reader.lastUID && (currentTime - reader.lastReadTime) < reader.DEBOUNCE_TIME) {
        return ""; // Bỏ qua thẻ trùng
    }
    
    // Cập nhật thông tin thẻ cuối
    reader.lastUID = uidString;
    reader.lastReadTime = currentTime;
    
    Serial.printf("[%s] Card detected: %s\n", reader.name.c_str(), uidString.c_str());
    return uidString;
}

// ====== RFID SETUP (giống thanhtoan) ======
void setup_rfid() {
    // Khởi tạo SPI với pins đúng (giống thanhtoan)
    SPI.begin(RC522_SCK, RC522_MISO, RC522_MOSI, -1);
    Serial.println("[SPI] Arduino SPI initialized");
    
    // Khởi tạo RFID A
    mfrc522_A.PCD_Init();
    byte versionA = mfrc522_A.PCD_ReadRegister(mfrc522_A.VersionReg);
    Serial.printf("[RFID] A Version: 0x%02X\n", versionA);
    
    // Khởi tạo RFID B
    mfrc522_B.PCD_Init();
    byte versionB = mfrc522_B.PCD_ReadRegister(mfrc522_B.VersionReg);
    Serial.printf("[RFID] B Version: 0x%02X\n", versionB);
    
    if (versionA == 0x00 || versionA == 0xFF) {
        Serial.println("[ERROR] RFID A not found! Check wiring.");
    } else {
        Serial.println("[RFID] A initialized successfully");
    }
    
    if (versionB == 0x00 || versionB == 0xFF) {
        Serial.println("[ERROR] RFID B not found! Check wiring.");
    } else {
        Serial.println("[RFID] B initialized successfully");
    }
}

// ====== RFID POLLING (thay thế RTOS tasks) ======
void checkRFIDCards() {
    // Check RFID A (IN) - chậm để giảm tiếng rít
    String cardA = readRFIDCard(rfid_A);
    if (cardA != "") {
        if (mqtt.connected()) {
            sendTriggerWithCard(rfid_A.trigger_type.c_str(), cardA.c_str());
        } else {
            Serial.println("[MQTT] Not connected, cannot send trigger");
        }
    }
    
    // Delay giữa 2 RFID (giảm xuống để nhanh hơn)
    delay(20);
    
    // Check RFID B (OUT)
    String cardB = readRFIDCard(rfid_B);
    if (cardB != "") {
        if (mqtt.connected()) {
            sendTriggerWithCard(rfid_B.trigger_type.c_str(), cardB.c_str());
        } else {
            Serial.println("[MQTT] Not connected, cannot send trigger");
        }
    }
}

// ====== SETUP ======
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n[SYSTEM] Parking System Starting...");

    wifiConnect();
    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    mqtt.setKeepAlive(15);  // Giảm keepalive để tránh timeout
    mqttConnect();

    setup_rfid();

    // ====== KHỞI TẠO RFID READERS ======
    rfid_A.device = &mfrc522_A;
    rfid_A.name = "RFID-A (IN)";
    rfid_A.trigger_type = "in";
    rfid_A.lastUID = "";
    rfid_A.lastReadTime = 0;

    rfid_B.device = &mfrc522_B;
    rfid_B.name = "RFID-B (OUT)";
    rfid_B.trigger_type = "out";
    rfid_B.lastUID = "";
    rfid_B.lastReadTime = 0;

    // ====== SETUP OTA ======
    setupOTA();

    Serial.println("[SYSTEM] All systems ready!");
}

// ====== LOOP (Arduino style, không RTOS) ======
void loop() {
    // Handle OTA
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

    // ✅ Thay thế RTOS tasks bằng polling đơn giản (như thanhtoan)
    checkRFIDCards();

    delay(50);  // ✅ Giống thanhtoan để nhanh và mượt
}
