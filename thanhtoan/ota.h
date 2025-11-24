// ota.h - OTA Configuration for Payment Terminal (with TFT display)
#ifndef OTA_H
#define OTA_H

#include <ArduinoOTA.h>
#include <WiFi.h>
#include <Adafruit_ST7735.h>
#include <ESPmDNS.h>
#include <esp_task_wdt.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Update.h>

// Khai báo extern để dùng tft từ main.cpp
extern Adafruit_ST7735 tft;

// Khai báo extern để set flag OTA từ main.cpp
extern bool isOTAUpdating;

// ====== OTA CONFIG ======
#define OTA_HOSTNAME "ESP32-Payment-Gate02"
#define OTA_PASSWORD "parking123"
#define OTA_PORT 3232

// ====== HTTP OTA CONFIG ======
#define OTA_SERVER_URL "http://192.168.1.165:5000"
#define DEVICE_TYPE "thanhtoan"
#define CURRENT_VERSION "v1.0.0"  // ✅ Thay đổi version này khi build firmware mới
#define OTA_CHECK_INTERVAL 3600000  // Check mỗi 1 giờ (milliseconds) - DISABLED

unsigned long lastOTACheck = 0;

// ====== HELPER FUNCTIONS - Display OTA Screens ======

// Helper: Vẽ header chung cho tất cả màn hình OTA
void drawOTAHeader(uint16_t borderColor, const char* title, uint16_t titleColor) {
    tft.fillScreen(ST77XX_BLACK);

    // Viền ngoài kép
    tft.drawRect(0, 0, 128, 160, borderColor);
    tft.drawRect(1, 1, 126, 158, borderColor);

    // Header bar
    tft.fillRect(3, 3, 122, 20, borderColor);

    // Title
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_BLACK);
    int titleX = (128 - strlen(title) * 6) / 2;  // Center text
    tft.setCursor(titleX, 9);
    tft.println(title);
}

// Màn hình bắt đầu OTA - REDESIGNED
void displayOTAStarting() {
    drawOTAHeader(ST77XX_ORANGE, "OTA UPDATE", ST77XX_BLACK);

    // Icon: Download arrow animation
    int centerX = 64;
    int centerY = 55;

    // Vẽ arrow xuống (animated)
    for (int stage = 0; stage < 3; stage++) {
        int offset = stage * 8;

        // Shaft
        tft.fillRect(61, 35 + offset, 6, 20, ST77XX_ORANGE);

        // Arrow head
        for (int i = 0; i < 8; i++) {
            tft.drawLine(64 - i, 55 + offset + i, 64 + i, 55 + offset + i, ST77XX_ORANGE);
        }

        delay(150);

        // Clear for next frame
        if (stage < 2) {
            tft.fillRect(55, 30 + offset, 18, 35, ST77XX_BLACK);
        }
    }

    // Warning text
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(10, 100);
    tft.println("DANG UPDATE");

    // Status text
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(18, 120);
    tft.println("Updating...");

    // Progress bar container (empty)
    tft.drawRect(10, 140, 108, 12, ST77XX_WHITE);
}

// Màn hình progress OTA - REDESIGNED (không chồng chữ)
void displayOTAProgress(unsigned int percent) {
    // ✅ FIX: Không vẽ lại toàn bộ, chỉ update progress bar

    // Clear progress bar area
    tft.fillRect(11, 141, 106, 10, ST77XX_BLACK);

    // Draw progress fill với gradient effect
    if (percent > 0) {
        int progressWidth = (int)(106 * percent / 100);

        // Main progress bar (green)
        tft.fillRect(11, 141, progressWidth, 10, ST77XX_GREEN);

        // Highlight effect (lighter green on top half)
        for (int i = 0; i < progressWidth; i += 2) {
            tft.drawPixel(11 + i, 141, 0x07E0);  // Lighter green
            tft.drawPixel(11 + i, 142, 0x07E0);
        }
    }

    // Clear và update percentage text
    tft.fillRect(30, 120, 70, 16, ST77XX_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_ORANGE);

    // Center percentage
    char buf[8];
    sprintf(buf, "%u%%", percent);
    int textWidth = strlen(buf) * 12;
    int textX = (128 - textWidth) / 2;

    tft.setCursor(textX, 120);
    tft.print(buf);
}

// Màn hình hoàn tất OTA - REDESIGNED
void displayOTAComplete() {
    drawOTAHeader(ST77XX_GREEN, "SUCCESS", ST77XX_BLACK);

    int centerX = 64;
    int centerY = 55;
    int radius = 25;

    // Animation: Vẽ circle fill từ trong ra ngoài
    for (int r = 0; r <= radius; r += 2) {
        tft.drawCircle(centerX, centerY, r, ST77XX_GREEN);
        delay(20);
    }

    // Fill solid circle
    tft.fillCircle(centerX, centerY, radius, ST77XX_GREEN);
    delay(150);

    // Vẽ checkmark animation (smooth)
    int check_x1 = 52, check_y1 = 55;
    int check_x2 = 60, check_y2 = 63;
    int check_x3 = 76, check_y3 = 47;

    // Left part of check
    for (int i = 0; i <= 8; i++) {
        int x = check_x1 + (check_x2 - check_x1) * i / 8;
        int y = check_y1 + (check_y2 - check_y1) * i / 8;
        tft.drawLine(check_x1, check_y1, x, y, ST77XX_BLACK);
        tft.drawLine(check_x1+1, check_y1, x+1, y, ST77XX_BLACK);
        tft.drawLine(check_x1, check_y1+1, x, y+1, ST77XX_BLACK);
        delay(20);
    }

    // Right part of check
    for (int i = 0; i <= 16; i++) {
        int x = check_x2 + (check_x3 - check_x2) * i / 16;
        int y = check_y2 + (check_y3 - check_y2) * i / 16;
        tft.drawLine(check_x2, check_y2, x, y, ST77XX_BLACK);
        tft.drawLine(check_x2+1, check_y2, x+1, y, ST77XX_BLACK);
        tft.drawLine(check_x2, check_y2+1, x, y+1, ST77XX_BLACK);
        delay(15);
    }

    delay(200);

    // Success message
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_GREEN);
    tft.setCursor(18, 100);
    tft.println("COMPLETE");

    // Reboot info
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(15, 125);
    tft.println("Rebooting in");

    tft.setTextColor(ST77XX_YELLOW);
    tft.setTextSize(2);
    tft.setCursor(54, 140);
    tft.println("3s");
}

// Màn hình lỗi OTA - REDESIGNED
void displayOTAError(String errorMsg) {
    drawOTAHeader(ST77XX_RED, "ERROR", ST77XX_BLACK);

    int centerX = 64;
    int centerY = 55;
    int radius = 25;

    // Animation: Vẽ circle fill từ trong ra ngoài (red)
    for (int r = 0; r <= radius; r += 2) {
        tft.drawCircle(centerX, centerY, r, ST77XX_RED);
        delay(15);
    }

    // Fill solid circle
    tft.fillCircle(centerX, centerY, radius, ST77XX_RED);
    delay(150);

    // Vẽ X mark animation (smooth)
    int xSize = 18;
    int x_left = centerX - xSize/2;
    int x_right = centerX + xSize/2;
    int y_top = centerY - xSize/2;
    int y_bottom = centerY + xSize/2;

    // Diagonal 1: top-left to bottom-right
    for (int i = 0; i <= xSize; i++) {
        int x = x_left + i;
        int y = y_top + i;
        tft.drawLine(x_left, y_top, x, y, ST77XX_BLACK);
        tft.drawLine(x_left+1, y_top, x+1, y, ST77XX_BLACK);
        tft.drawLine(x_left, y_top+1, x, y+1, ST77XX_BLACK);
        delay(10);
    }

    // Diagonal 2: top-right to bottom-left
    for (int i = 0; i <= xSize; i++) {
        int x = x_right - i;
        int y = y_top + i;
        tft.drawLine(x_right, y_top, x, y, ST77XX_BLACK);
        tft.drawLine(x_right-1, y_top, x-1, y, ST77XX_BLACK);
        tft.drawLine(x_right, y_top+1, x, y+1, ST77XX_BLACK);
        delay(10);
    }

    delay(200);

    // Error message
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 100);
    tft.println("FAILED");

    // Error detail
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_YELLOW);
    tft.setCursor(5, 125);
    tft.print("Error: ");

    tft.setTextColor(ST77XX_WHITE);
    // Wrap long error messages
    if (errorMsg.length() > 15) {
        tft.setCursor(5, 137);
        tft.println(errorMsg.substring(0, 15));
        tft.setCursor(5, 147);
        tft.println(errorMsg.substring(15));
    } else {
        tft.setCursor(5, 137);
        tft.println(errorMsg);
    }
}

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

    // Set flag để ngăn state machine vẽ đè lên màn hình OTA
    isOTAUpdating = true;

    // Show OTA starting screen on TFT
    displayOTAStarting();
    disableCore0WDT();
    disableCore1WDT();

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
                uint8_t buff[128] = { 0 };
                size_t written = 0;
                unsigned int lastPercent = 0;

                while (http.connected() && (contentLength > 0 || contentLength == -1)) {
                    size_t size = stream->available();

                    if (size) {
                        int c = stream->readBytes(buff, ((size > sizeof(buff)) ? sizeof(buff) : size));
                        Update.write(buff, c);
                        written += c;

                        if (contentLength > 0) {
                            contentLength -= c;
                            unsigned int percent = (written * 100) / http.getSize();

                            // Update TFT every 5%
                            if (percent != lastPercent && percent % 5 == 0) {
                                displayOTAProgress(percent);
                                lastPercent = percent;
                            }
                        }

                        esp_task_wdt_reset();
                    }
                    delay(1);
                }

                if (Update.end()) {
                    if (Update.isFinished()) {
                        Serial.println("[HTTP OTA] Update successfully completed. Rebooting...");
                        displayOTAComplete();
                        http.end();
                        delay(3000);
                        ESP.restart();
                        return true;
                    } else {
                        Serial.println("[HTTP OTA] Update not finished? Something went wrong!");
                        displayOTAError("Not finished");
                        enableCore0WDT();
                        enableCore1WDT();
                        isOTAUpdating = false;  // Reset flag
                        return false;
                    }
                } else {
                    Serial.printf("[HTTP OTA] Error Occurred. Error #: %d\n", Update.getError());
                    displayOTAError("Update failed");
                    enableCore0WDT();
                    enableCore1WDT();
                    isOTAUpdating = false;  // Reset flag
                    return false;
                }
            } else {
                Serial.println("[HTTP OTA] Not enough space to begin OTA");
                displayOTAError("No space");
                http.end();
                enableCore0WDT();
                enableCore1WDT();
                isOTAUpdating = false;  // Reset flag
                return false;
            }
        } else {
            Serial.println("[HTTP OTA] Content length is 0");
            displayOTAError("Empty file");
            http.end();
            enableCore0WDT();
            enableCore1WDT();
            isOTAUpdating = false;  // Reset flag
            return false;
        }
    } else {
        Serial.printf("[HTTP OTA] Download failed, HTTP code: %d\n", httpCode);
        displayOTAError("Download fail");
        http.end();
        enableCore0WDT();
        enableCore1WDT();
        isOTAUpdating = false;  // Reset flag
        return false;
    }

    http.end();
    enableCore0WDT();
    enableCore1WDT();
    isOTAUpdating = false;  // Reset flag
    return false;
}

void checkAndUpdateHTTPOTA() {
    // ❌ DISABLED: Periodic auto-check (only manual update via MQTT trigger)
    // if (millis() - lastOTACheck > OTA_CHECK_INTERVAL) {
    //     Serial.println("[HTTP OTA] Periodic update check...");
    //     if (checkForHTTPUpdate()) {
    //         Serial.println("[HTTP OTA] Update available! Starting download...");
    //         performHTTPUpdate();
    //     }
    //     lastOTACheck = millis();
    // }
}

void setupOTA() {
    Serial.println("[OTA] Setting up OTA...");

    // Đợi WiFi stable trước khi setup mDNS
    delay(500);

    // Kiểm tra WiFi trước khi setup mDNS
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[OTA] ERROR: WiFi not connected! Cannot setup mDNS.");
        return;
    }

    // Khởi tạo mDNS - RẤT QUAN TRỌNG để PlatformIO tìm được thiết bị
    if (!MDNS.begin(OTA_HOSTNAME)) {
        Serial.println("[OTA] ERROR: mDNS init failed!");
        // Không return - vẫn cho phép OTA qua IP
    } else {
        Serial.printf("[OTA] mDNS started: %s.local\n", OTA_HOSTNAME);
        MDNS.addService("arduino", "tcp", OTA_PORT);
    }

    ArduinoOTA.setHostname(OTA_HOSTNAME);
    ArduinoOTA.setPassword(OTA_PASSWORD);
    ArduinoOTA.setPort(OTA_PORT);

    // Callback khi bắt đầu update
    ArduinoOTA.onStart([]() {
        String type;
        if (ArduinoOTA.getCommand() == U_FLASH) {
            type = "sketch";
        } else {  // U_SPIFFS
            type = "filesystem";
        }
        Serial.println("[OTA] Start updating " + type);

        // Set flag để ngăn state machine vẽ đè lên màn hình OTA
        isOTAUpdating = true;

        // Hiển thị màn hình bắt đầu OTA
        displayOTAStarting();

        // Tắt watchdog timer để tránh reset trong OTA
        disableCore0WDT();
        disableCore1WDT();
    });

    // Callback trong quá trình update (hiển thị progress)
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        unsigned int percent = (progress / (total / 100));
        Serial.printf("[OTA] Progress: %u%%\r", percent);

        // ✅ FIX: Cập nhật progress bar trên TFT mỗi 1% để hiển thị mượt hơn
        static unsigned int lastPercent = 0;
        if (percent != lastPercent) {
            displayOTAProgress(percent);
            lastPercent = percent;
        }

        // Feed watchdog để tránh timeout trong quá trình OTA
        esp_task_wdt_reset();
    });

    // Callback khi update xong
    ArduinoOTA.onEnd([]() {
        Serial.println("\n[OTA] Update complete!");
        displayOTAComplete();
        delay(3000);

        // Bật lại watchdog
        enableCore0WDT();
        enableCore1WDT();

        // Reset flag (sẽ restart nên không cần thiết, nhưng để an toàn)
        isOTAUpdating = false;
    });

    // Callback khi có lỗi
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("[OTA] Error[%u]: ", error);
        String errorMsg;

        if (error == OTA_AUTH_ERROR) errorMsg = "Auth Failed";
        else if (error == OTA_BEGIN_ERROR) errorMsg = "Begin Failed";
        else if (error == OTA_CONNECT_ERROR) errorMsg = "Connect";
        else if (error == OTA_RECEIVE_ERROR) errorMsg = "Receive";
        else if (error == OTA_END_ERROR) errorMsg = "End Failed";
        else errorMsg = "Unknown Error";

        Serial.println(errorMsg);
        displayOTAError(errorMsg);
        delay(3000);

        // Bật lại watchdog
        enableCore0WDT();
        enableCore1WDT();

        // Reset flag để quay lại hoạt động bình thường
        isOTAUpdating = false;
    });

    ArduinoOTA.begin();
    Serial.println("[OTA] ArduinoOTA ready!");
    Serial.printf("[OTA] Hostname: %s\n", OTA_HOSTNAME);
    Serial.printf("[OTA] IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[OTA] Current version: %s\n", CURRENT_VERSION);
    Serial.println("[HTTP OTA] Auto-update DISABLED. Use web UI to trigger manual updates.");
    lastOTACheck = millis();
}

#endif
