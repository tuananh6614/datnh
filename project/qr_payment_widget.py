"""
QR Code Payment Display Widget
Phiên bản VietQR:
1. Sử dụng img.vietqr.io API để tạo QR code chuẩn VietQR (hỗ trợ VietinBank)
2. Simplify resize logic - sử dụng KeepAspectRatio để QR không bị méo
3. Xóa thông tin thời gian - giao diện thoáng hơn
"""

import io
import logging
import base64
from datetime import datetime
from decimal import Decimal
from typing import Optional

import qrcode
from PIL import Image
import requests  # Thêm requests để load ảnh từ URL nếu cần

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGroupBox
)

from sepay_helper import SEPAY_CONFIG


class QRPaymentWidget(QDialog):
    """Dialog hiển thị QR thanh toán"""

    payment_cancelled = Signal()  # Signal khi hủy thanh toán

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qr_data = None
        self.payment_info = {}
        self.timeout_seconds = 60
        self.timer_timeout = QTimer(self)
        self.timer_timeout.setSingleShot(True)
        self.timer_timeout.timeout.connect(self.on_timeout)

        self._build_ui()

    def _build_ui(self):
        """Xây dựng giao diện"""
        self.setWindowTitle("Thanh toán - Quét mã QR")
        self.setModal(True)

        # ✅ SỬA: Tăng chiều cao để hiển thị QR to hơn (xóa bỏ thông tin thời gian)
        self.setMinimumWidth(600)
        self.setMinimumHeight(800)

        # ✅ Status header
        self.lbl_status_icon = QLabel("⌛")
        font_status_icon = QFont()
        font_status_icon.setPointSize(20)
        self.lbl_status_icon.setFont(font_status_icon)

        self.lbl_status_text = QLabel("Đang chờ thanh toán...")
        font_status = QFont()
        font_status.setPointSize(15)
        font_status.setBold(True)
        self.lbl_status_text.setFont(font_status)
        self.lbl_status_text.setAlignment(Qt.AlignCenter)

        status_layout = QHBoxLayout()
        status_layout.addStretch(1)
        status_layout.addWidget(self.lbl_status_icon)
        status_layout.addSpacing(8)
        status_layout.addWidget(self.lbl_status_text)
        status_layout.addStretch(1)

        # ✅ QR Code display - FIXED SIZE SQUARE
        self.lbl_qr = QLabel()
        self.lbl_qr.setAlignment(Qt.AlignCenter)

        # ✅ QUAN TRỌNG: Ép QR thành hình vuông 330x330 để không bị méo
        self.lbl_qr.setFixedSize(330, 330)
        self.lbl_qr.setScaledContents(False)

        # ✅ Nền trắng xung quanh QR code
        palette_qr = QPalette()
        palette_qr.setColor(QPalette.Window, QColor(255, 255, 255))
        self.lbl_qr.setPalette(palette_qr)
        self.lbl_qr.setAutoFillBackground(True)
        self.lbl_qr.setContentsMargins(0, 0, 0, 0)

        # ✅ Container cho QR - màu trắng, bo góc
        qr_bg = QWidget()
        qr_bg.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
        """)
        qr_bg_layout = QVBoxLayout(qr_bg)
        qr_bg_layout.addWidget(self.lbl_qr, 0, Qt.AlignCenter)
        qr_bg_layout.setContentsMargins(10, 10, 10, 10)
        self.qr_bg_widget = qr_bg

        # ✅ Payment info
        self.lbl_plate = QLabel()
        font_plate = QFont()
        font_plate.setPointSize(18)
        font_plate.setBold(True)
        self.lbl_plate.setFont(font_plate)
        palette_plate = QPalette()
        palette_plate.setColor(QPalette.WindowText, QColor(51, 51, 51))
        self.lbl_plate.setPalette(palette_plate)

        self.lbl_amount = QLabel()
        font_amount = QFont()
        font_amount.setPointSize(16)
        font_amount.setBold(True)
        self.lbl_amount.setFont(font_amount)
        palette_amount = QPalette()
        palette_amount.setColor(QPalette.WindowText, QColor(220, 53, 69))  # red
        self.lbl_amount.setPalette(palette_amount)


        self.lbl_instruction = QLabel(
            "Vui lòng quét QR và chuyển khoản"
        )
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        font_instr = QFont()
        font_instr.setPointSize(14)
        self.lbl_instruction.setFont(font_instr)
        palette_instr = QPalette()
        palette_instr.setColor(QPalette.WindowText, QColor(13, 110, 253))  # blue
        self.lbl_instruction.setPalette(palette_instr)

        # ✅ Timeout warning
        self.lbl_timeout_warning = QLabel()
        self.lbl_timeout_warning.setAlignment(Qt.AlignCenter)
        font_warn = QFont()
        font_warn.setPointSize(12)
        self.lbl_timeout_warning.setFont(font_warn)
        palette_warn = QPalette()
        palette_warn.setColor(QPalette.WindowText, QColor(220, 53, 69))  # red
        self.lbl_timeout_warning.setPalette(palette_warn)
        self.lbl_timeout_warning.hide()

        # ✅ Bank info - HTML formatting
        self.lbl_bank_name = QLabel("Ngân hàng: —")
        self.lbl_bank_name.setAlignment(Qt.AlignCenter)

        self.lbl_account_number = QLabel("Số TK: —")
        self.lbl_account_number.setAlignment(Qt.AlignCenter)
        self.lbl_account_number.setStyleSheet("color: #cc0000; font-size: 14px; font-weight: bold;")

        self.lbl_account_holder = QLabel("Tên: —")
        self.lbl_account_holder.setAlignment(Qt.AlignCenter)

        bank_info_layout = QVBoxLayout()
        bank_info_layout.setSpacing(6)
        bank_info_layout.setContentsMargins(12, 8, 12, 8)
        bank_info_layout.addWidget(self.lbl_bank_name, 0, Qt.AlignCenter)
        bank_info_layout.addWidget(self.lbl_account_number, 0, Qt.AlignCenter)
        bank_info_layout.addWidget(self.lbl_account_holder, 0, Qt.AlignCenter)

        bank_info_group = QWidget()
        bank_info_group.setLayout(bank_info_layout)

        # ✅ Button
        font_btn = QFont()
        font_btn.setPointSize(14)
        font_btn.setBold(True)

        btn_cancel = QPushButton("✗ Hủy thanh toán")
        btn_cancel.setMinimumHeight(45)
        btn_cancel.setFont(font_btn)
        btn_cancel.setStyleSheet("QPushButton{background:#dc3545;color:#fff;border:none;padding:8px 16px;border-radius:6px;} QPushButton:pressed{background:#a71d2a;}")
        btn_cancel.clicked.connect(self.on_cancel)

        # ✅ Layout
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Biển số xe:"))
        info_layout.addWidget(self.lbl_plate)
        info_layout.addWidget(QLabel("Số tiền:"))
        info_layout.addWidget(self.lbl_amount)

        info_group = QGroupBox("Thông tin thanh toán")
        info_group.setLayout(info_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(btn_cancel, 0, Qt.AlignCenter)
        button_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(info_group)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.lbl_instruction)

        # ✅ Thêm QR code container vào giữa
        qr_center_layout = QHBoxLayout()
        qr_center_layout.addStretch(1)
        qr_center_layout.addWidget(self.qr_bg_widget, 0, Qt.AlignCenter)
        qr_center_layout.addStretch(1)
        main_layout.addLayout(qr_center_layout, 1)
        
        main_layout.addWidget(bank_info_group)
        main_layout.addWidget(self.lbl_timeout_warning)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _set_qr_pixmap(self, pixmap: QPixmap):
        """Helper để set pixmap vào khung vuông 330x330"""
        if pixmap and not pixmap.isNull():
            # ✅ ĐƠNGIẢN: Scale ảnh vào khung 330x330 với KeepAspectRatio
            scaled_pixmap = pixmap.scaled(
                330, 330,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.lbl_qr.setPixmap(scaled_pixmap)
        else:
            self.lbl_qr.setText("❌ Lỗi hiển thị ảnh")

    @staticmethod
    def _format_datetime_display(value) -> str:
        """Chuẩn hóa giá trị thời gian sang định dạng DD/MM/YYYY HH:MM:SS."""
        if value in (None, "", 0):
            return "Không rõ"

        dt_obj = None

        if isinstance(value, datetime):
            dt_obj = value
        elif isinstance(value, (int, float)):
            try:
                dt_obj = datetime.fromtimestamp(value)
            except Exception:
                dt_obj = None
        elif isinstance(value, str):
            candidate = value.strip()
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                dt_obj = datetime.fromisoformat(candidate)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                    try:
                        dt_obj = datetime.strptime(candidate, fmt)
                        break
                    except ValueError:
                        continue

        if dt_obj:
            return dt_obj.strftime("%d/%m/%Y %H:%M:%S")

        return str(value)

    @staticmethod
    def _pixmap_from_bytes(data: bytes) -> Optional[QPixmap]:
        """Tạo QPixmap trực tiếp từ bytes PNG/JPG."""
        qimage = QImage.fromData(data)
        if qimage.isNull():
            return None
        return QPixmap.fromImage(qimage)

    @staticmethod
    def _pixmap_from_pil(img: Image.Image) -> Optional[QPixmap]:
        """Chuyển PIL Image sang QPixmap thông qua buffer PNG."""
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return QRPaymentWidget._pixmap_from_bytes(buffer.getvalue())

    @staticmethod
    def _normalize_amount(value) -> int:
        """Chuẩn hóa số tiền về int không âm.
        Handles int, float, Decimal (from PostgreSQL), and string types.
        """
        try:
            if value is None:
                return 0
            # ✅ FIX: Include Decimal type from PostgreSQL
            if isinstance(value, (int, float, Decimal)):
                return max(0, int(value))
            if isinstance(value, str):
                cleaned = value.replace(",", "").strip()
                if not cleaned:
                    return 0
                return max(0, int(float(cleaned)))
        except (ValueError, TypeError):
            logging.warning(f"[QR] Invalid amount value: {value}")
        return 0

    def display_payment(self, payment_data: dict):
        """
        Hiển thị QR thanh toán
        """
        try:
            self.payment_info = payment_data
            logging.info(f"[QR WIDGET] Displaying payment: {payment_data.get('plate_number')}")

            # ✅ Hiển thị thông tin text
            plate = payment_data.get('plate_number', 'N/A')
            amount = self._normalize_amount(payment_data.get('amount', 0))
            payment_data['amount'] = amount 

            self.lbl_plate.setText(f"{plate}")
            self.lbl_amount.setText(f"{amount:,} VNĐ")

            bank_display_name = (
                payment_data.get('bank_display_name')
                or payment_data.get('bank_name')
                or SEPAY_CONFIG.get("bank_display_name")
                or SEPAY_CONFIG.get("bank_short_name")
                or "Không rõ"
            )
            account_number = payment_data.get('account_number') or SEPAY_CONFIG.get("account_number") or "—"
            account_holder = payment_data.get('account_name') or SEPAY_CONFIG.get("account_name") or "—"

            # ✅ HTML formatting cho bank info
            self.lbl_bank_name.setText(f"Ngân hàng: <b>{bank_display_name}</b>")
            self.lbl_account_number.setText(f"Số TK: <b>{account_number}</b>")
            self.lbl_account_holder.setText(f"Tên: {account_holder}")

            self.lbl_status_icon.setText("🔵")
            self.lbl_status_text.setText("Đang chờ thanh toán...")

            # ✅ XỬ LÝ HIỂN THỊ ẢNH QR (Logic quan trọng nhất)
            # FIX: Ưu tiên tạo QR gốc từ content để tránh lỗi logo che mã
            qr_displayed = False

            # Cách 1 (Ưu tiên): Tạo QR thủ công từ content (sẽ ra hình vuông, không logo)
            qr_content = payment_data.get('qr_content')
            if qr_content and qr_content.strip():
                try:
                    logging.info(f"[QR] Creating QR from content (priority)")
                    qr = qrcode.QRCode(
                        version=5,
                        # Tăng mức độ sửa lỗi để đảm bảo quét được
                        error_correction=qrcode.constants.ERROR_CORRECT_H,
                        box_size=10,
                        border=2,
                    )
                    qr.add_data(qr_content)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    pixmap = self._pixmap_from_pil(img)
                    if pixmap:
                        self._set_qr_pixmap(pixmap)
                        qr_displayed = True
                        logging.info("[QR] ✅ Displayed QR from VietQR content (no logo)")
                except Exception as e:
                    logging.warning(f"[QR] ⚠️ Failed to create QR from content: {e}")

            # Cách 2: Hiển thị PNG base64 từ Sepay (Fallback)
            if not qr_displayed:
                qr_base64 = payment_data.get('qr_base64')
                if qr_base64 and len(qr_base64) > 100:
                    try:
                        img_data = base64.b64decode(qr_base64)
                        pixmap = self._pixmap_from_bytes(img_data)
                        if pixmap:
                            self._set_qr_pixmap(pixmap)
                            qr_displayed = True
                            logging.info("[QR] ✅ Displayed base64 PNG from Sepay (fallback)")
                    except Exception as e:
                        logging.warning(f"[QR] ⚠️ Failed to display base64 PNG: {e}")

            # Cách 3: Download từ URL nếu các cách trên fail
            if not qr_displayed:
                qr_url = payment_data.get('qr_url')
                if qr_url:
                    try:
                        logging.info(f"[QR] Downloading QR from URL: {qr_url}")
                        response = requests.get(qr_url, timeout=5)
                        if response.status_code == 200:
                            pixmap = self._pixmap_from_bytes(response.content)
                            if pixmap:
                                self._set_qr_pixmap(pixmap)
                                qr_displayed = True
                                logging.info("[QR] ✅ Displayed QR from URL")
                    except Exception as e:
                        logging.warning(f"[QR] ⚠️ Failed to download QR from URL: {e}")

            # Nếu tất cả cách fail
            if not qr_displayed:
                self.lbl_qr.setText("❌ Lỗi hiển thị QR\nVui lòng kiểm tra kết nối Internet")

            # Reset timeout
            self.timer_timeout.stop()
            self.timer_timeout.start(self.timeout_seconds * 1000)
            self.lbl_timeout_warning.hide()

            # Hiển thị dialog
            if not self.isVisible():
                self.setWindowModality(Qt.ApplicationModal)
                self.show()
            self.raise_()
            self.activateWindow()

        except Exception as e:
            logging.error(f"[QR] Error displaying payment: {e}", exc_info=True)
            self.close()

    def on_cancel(self):
        """Hủy thanh toán"""
        logging.info(f"[QR] Payment cancelled for session {self.payment_info.get('session_id')}")
        self.payment_cancelled.emit()
        self.timer_timeout.stop()
        self.reject()

    def on_timeout(self):
        """Timeout - tự động đóng"""
        logging.warning("[QR] Payment timeout - closing dialog")
        self.timer_timeout.stop()
        self.reject()


class QRDisplayLabel(QLabel):
    """Label nhỏ gọn để hiển thị QR tại một góc màn hình"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # Tắt auto scale
        self.setScaledContents(False)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))  # white background
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setContentsMargins(5, 5, 5, 5)
        self.setMaximumWidth(350)
        self.setMaximumHeight(380)

    def set_qr_from_content(self, qr_content: str):
        """Tạo QR từ content"""
        try:
            qr = qrcode.QRCode(
                version=4,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=1,
            )
            qr.add_data(qr_content)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            pixmap = QRPaymentWidget._pixmap_from_pil(img)
            if pixmap:
                # Scale giữ tỉ lệ
                self.setPixmap(pixmap.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logging.error(f"[QR] Error creating QR: {e}")

    def set_qr_from_base64(self, qr_base64: str):
        """Tạo QR từ base64 PNG"""
        try:
            img_data = base64.b64decode(qr_base64)
            pixmap = QRPaymentWidget._pixmap_from_bytes(img_data)
            if pixmap:
                # Scale giữ tỉ lệ
                self.setPixmap(pixmap.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logging.error(f"[QR] Error displaying base64: {e}")