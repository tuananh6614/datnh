// ota.h - OTA Configuration for Parking Gate (baidoxe)
#ifndef OTA_H
#define OTA_H

#include <ArduinoOTA.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Update.h>

// ====== OTA CONFIG ======
#define OTA_HOSTNAME "ESP32-Parking-Gate01"
#define OTA_PASSWORD "parking123"
#define OTA_PORT 3232

// ====== HTTP OTA CONFIG ======
#define OTA_SERVER_URL "http://192.168.1.165:5000"
#define DEVICE_TYPE "baidoxe"
#define CURRENT_VERSION "v1.0.0"  // ✅ Thay đổi version này khi build firmware mới
#define OTA_CHECK_INTERVAL 3600000  // Check mỗi 1 giờ (milliseconds)

unsigned long lastOTACheck = 0;

// ====== HTTP OTA FUNCTIONS ======
bool checkForHTTPUpdate() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[HTTP OTA] WiFi not connected");
        return false;
    }

    HTTPClient http;
    String checkUrl = String(OTA_SERVER_URL) + "/api/ota/check/" + DEVICE_TYPE;

    Serial.printf("[HTTP OTA] Checking for updates: %s\n", checkUrl.c_str());
    http.begin(checkUrl);
    http.setTimeout(5000);

    int httpCode = http.GET();

    if (httpCode == 200) {
        String payload = http.getString();
        Serial.printf("[HTTP OTA] Response: %s\n", payload.c_str());

        JsonDocument doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (error) {
            Serial.printf("[HTTP OTA] JSON parse error: %s\n", error.c_str());
            http.end();
            return false;
        }

        bool available = doc["available"].as<bool>();
        String newVersion = doc["version"].as<String>();

        http.end();

        if (available && newVersion != CURRENT_VERSION) {
            Serial.printf("[HTTP OTA] New version available: %s (current: %s)\n",
                         newVersion.c_str(), CURRENT_VERSION);
            return true;
        } else {
            Serial.printf("[HTTP OTA] Already up to date (v%s)\n", CURRENT_VERSION);
            return false;
        }
    } else {
        Serial.printf("[HTTP OTA] Check failed, HTTP code: %d\n", httpCode);
        http.end();
        return false;
    }
}

bool performHTTPUpdate() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[HTTP OTA] WiFi not connected");
        return false;
    }

    HTTPClient http;
    String downloadUrl = String(OTA_SERVER_URL) + "/api/ota/download/" + DEVICE_TYPE;

    Serial.printf("[HTTP OTA] Downloading firmware: %s\n", downloadUrl.c_str());
    http.begin(downloadUrl);
    http.setTimeout(30000);  // 30 seconds timeout

    int httpCode = http.GET();

    if (httpCode == 200) {
        int contentLength = http.getSize();
        Serial.printf("[HTTP OTA] Firmware size: %d bytes\n", contentLength);

        if (contentLength > 0) {
            bool canBegin = Update.begin(contentLength);

            if (canBegin) {
                Serial.println("[HTTP OTA] Starting update...");

                WiFiClient *stream = http.getStreamPtr();
                size_t written = Update.writeStream(*stream);

                if (written == contentLength) {
                    Serial.println("[HTTP OTA] Written : " + String(written) + " successfully");
                } else {
                    Serial.println("[HTTP OTA] Written only : " + String(written) + "/" + String(contentLength));
                }

                if (Update.end()) {
                    if (Update.isFinished()) {
                        Serial.println("[HTTP OTA] Update successfully completed. Rebooting...");
                        http.end();
                        delay(1000);
                        ESP.restart();
                        return true;
                    } else {
                        Serial.println("[HTTP OTA] Update not finished? Something went wrong!");
                        return false;
                    }
                } else {
                    Serial.printf("[HTTP OTA] Error Occurred. Error #: %d\n", Update.getError());
                    return false;
                }
            } else {
                Serial.println("[HTTP OTA] Not enough space to begin OTA");
                http.end();
                return false;
            }
        } else {
            Serial.println("[HTTP OTA] Content length is 0");
            http.end();
            return false;
        }
    } else {
        Serial.printf("[HTTP OTA] Download failed, HTTP code: %d\n", httpCode);
        http.end();
        return false;
    }

    http.end();
    return false;
}

void checkAndUpdateHTTPOTA() {
    if (millis() - lastOTACheck > OTA_CHECK_INTERVAL) {
        Serial.println("[HTTP OTA] Periodic update check...");

        if (checkForHTTPUpdate()) {
            Serial.println("[HTTP OTA] Update available! Starting download...");
            performHTTPUpdate();
        }

        lastOTACheck = millis();
    }
}

// ====== ARDUINO OTA (WiFi Local) ======
void setupOTA() {
    Serial.println("[OTA] Setting up OTA...");

    // Đợi WiFi stable
    delay(500);

    // Kiểm tra WiFi
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[OTA] ERROR: WiFi not connected!");
        return;
    }

    // Khởi tạo mDNS
    if (!MDNS.begin(OTA_HOSTNAME)) {
        Serial.println("[OTA] ERROR: mDNS init failed!");
    } else {
        Serial.printf("[OTA] mDNS started: %s.local\n", OTA_HOSTNAME);
        MDNS.addService("arduino", "tcp", OTA_PORT);
    }

    ArduinoOTA.setHostname(OTA_HOSTNAME);
    ArduinoOTA.setPassword(OTA_PASSWORD);
    ArduinoOTA.setPort(OTA_PORT);

    ArduinoOTA.onStart([]() {
        Serial.println("[OTA] Start updating...");
    });

    ArduinoOTA.onEnd([]() {
        Serial.println("\n[OTA] Update complete!");
    });

    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("[OTA] Progress: %u%%\r", (progress / (total / 100)));
    });

    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("[OTA] Error[%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");
    });

    ArduinoOTA.begin();
    Serial.println("[OTA] ArduinoOTA ready!");
    Serial.printf("[OTA] Hostname: %s\n", OTA_HOSTNAME);
    Serial.printf("[OTA] IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[OTA] Current version: %s\n", CURRENT_VERSION);

    // ❌ DISABLED: Auto-check on boot (only manual update via MQTT trigger)
    // Serial.println("[HTTP OTA] Checking for updates on boot...");
    // delay(2000);  // Wait for network to stabilize
    // if (checkForHTTPUpdate()) {
    //     Serial.println("[HTTP OTA] Update available on boot! Starting download...");
    //     performHTTPUpdate();
    // }
    lastOTACheck = millis();
}

#endif
