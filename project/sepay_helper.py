"""
SEPAY HELPER - ĐÚNG THEO TÀI LIỆU

- QR Code động từ qr.sepay.vn
- API verification với Bearer token
- Webhook processing
"""

import base64
import json
import logging
import urllib.parse
from typing import Dict, Optional

import requests

from sepay_config import SEPAY_CONFIG, SEPAY_ENDPOINTS

logger = logging.getLogger(__name__)


class SePay:
    """Minimal helper around SePay public APIs."""

    def __init__(self):
        self.api_token = SEPAY_CONFIG['api_token']
        self.api_url = SEPAY_CONFIG['api_url']
        self.qr_url = SEPAY_CONFIG['qr_url']
        self.account_number = SEPAY_CONFIG['account_number']
        self.account_name = SEPAY_CONFIG['account_name']
        self.bank_short_name = SEPAY_CONFIG['bank_short_name']

    def _make_api_request(self, endpoint, params=None):
        """Gửi GET request đến SePay API với Bearer token"""
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }

        url = self.api_url + endpoint

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error("SePay API Error: %s - %s", response.status_code, response.text)
                return {'status': response.status_code, 'error': response.text}

        except Exception as e:
            logger.exception("SePay API Error: %s", e)
            return {'status': 500, 'error': str(e)}

    def generate_qr_url(self, amount, description):
        """
        Tạo QR Code từ VietQR v2 API
        Trả về URL ảnh QR Code
        """
        payload = {
            "accountNo": self.account_number,
            "accountName": self.account_name,
            "acqId": "970415",  # VietinBank ACQ ID
            "amount": amount,
            "addInfo": description,
            "format": "text",
            "template": "compact"
        }
        
        try:
            response = requests.post(
                self.qr_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "00":
                    return data["data"]["qrDataURL"]  # Trả về data URL của QR image
                else:
                    logger.error("VietQR API error: %s", data.get("desc"))
                    return None
            else:
                logger.error("VietQR HTTP error: %d", response.status_code)
                return None
                
        except Exception as e:
            logger.error("VietQR request error: %s", e)
            return None

    def download_qr_png(self, qr_data_url):
        """
        Xử lý QR data URL từ VietQR API

        Returns:
            str: Base64 encoded PNG, hoặc None nếu lỗi
        """
        try:
            if qr_data_url and qr_data_url.startswith("data:image/"):
                # Đây là data URL, extract base64 part
                if "base64," in qr_data_url:
                    base64_data = qr_data_url.split("base64,")[1]
                    logger.info("✅ [VIETQR] Extracted base64 from data URL: %d chars", len(base64_data))
                    return base64_data
                else:
                    logger.error("❌ [VIETQR] Invalid data URL format")
                    return None
            else:
                # Fallback: try to download as regular URL
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "image/png,image/*,*/*"
                }

                response = requests.get(qr_data_url, headers=headers, timeout=15)

                if response.status_code == 200:
                    png_data = response.content
                    base64_data = base64.b64encode(png_data).decode("utf-8")
                    logger.info("✅ [VIETQR] Downloaded PNG: %d bytes → Base64: %d chars", len(png_data), len(base64_data))
                    return base64_data
                else:
                    logger.error("❌ [VIETQR] PNG download failed: %s", response.status_code)
                    return None

        except Exception as e:
            logger.exception("❌ [VIETQR] PNG processing error: %s", e)
            return None

    def verify_payment(self, amount, description, limit=10):
        """
        Verify payment bằng cách check API transactions

        Args:
            amount (int): Số tiền cần verify
            description (str): Nội dung giao dịch
            limit (int): Số transaction gần nhất để check

        Returns:
            dict: Transaction info nếu tìm thấy, None nếu không
        """
        params = {
            'limit': limit,
            'amount_in': amount
        }

        logger.info(f"[SePay] Verifying payment: amount={amount}, description='{description}'")
        response = self._make_api_request(SEPAY_ENDPOINTS['transactions_list'], params)
        logger.info(f"[SePay] API response status: {response.get('status')}, transactions: {len(response.get('transactions', []))}")

        if response.get('status') == 200:
            transactions = response.get('transactions', [])

            # Tìm transaction có nội dung khớp
            for tx in transactions:
                tx_amount = tx.get('amount_in')
                tx_content = tx.get('transaction_content', '')
                logger.debug(f"[SePay] Checking TX: amount={tx_amount}, content='{tx_content}'")

                if (tx_amount == str(amount) and
                    description.upper() in tx_content.upper()):
                    logger.info(f"[SePay] ✅ MATCH FOUND: {tx}")
                    return tx

        logger.info(f"[SePay] ❌ No matching transaction found")
        return None

    def get_recent_transactions(self, limit=20):
        """Lấy danh sách giao dịch gần đây"""
        params = {'limit': limit}
        return self._make_api_request(SEPAY_ENDPOINTS['transactions_list'], params)

    def get_bank_accounts(self):
        """Lấy danh sách tài khoản ngân hàng"""
        return self._make_api_request(SEPAY_ENDPOINTS['bankaccounts_list'])


# Global instance
sepay = SePay()


def create_parking_payment_sepay(session_id, amount):
    """
    ✅ ĐÚNG: Sử dụng VietQR API (img.vietqr.io) - hỗ trợ VietinBank

    Returns:
        dict: {
            'success': bool,
            'qr_url': str,        # URL QR image từ VietQR (VietinBank hỗ trợ)
            'qr_content': str,    # DEPRECATED - để trống
            'qr_base64': str,     # Base64 encoded PNG
            'transaction_id': str,
            'amount': int,
            'session_id': int
        }
    """
    try:
        # Nội dung chuyển khoản cho VietinBank - PHẢI có prefix SEVQR
        description = f"SEVQR BAI XE SESSION {session_id}"
        transaction_id = f"PARK_{session_id}_{amount}"

        # Tạo QR URL từ VietQR API (img.vietqr.io)
        qr_url = sepay.generate_qr_url(amount, description)

        # ✅ Tải PNG và encode base64
        qr_base64 = sepay.download_qr_png(qr_url)

        if not qr_base64:
            logger.error("❌ [VIETQR] Failed to download PNG for session %s", session_id)
            return {
                'success': False,
                'error': 'Failed to download QR PNG from Sepay'
            }

        logger.info("✅ [VIETQR] QR generated: %s, Amount: %s", transaction_id, f"{amount:,}")
        logger.info("✅ [VIETQR] QR URL: %s", qr_url)
        logger.info("✅ [VIETQR] Base64 length: %d chars", len(qr_base64))

        return {
            'success': True,
            'qr_url': qr_url,                           # URL ảnh QR từ VietQR (VietinBank hỗ trợ)
            'qr_content': '',                           # DEPRECATED - bỏ trống
            'qr_base64': qr_base64,                     # Base64 PNG cho ESP32
            'transaction_id': transaction_id,
            'amount': amount,
            'session_id': session_id,
            'bank_name': SEPAY_CONFIG['bank_short_name'],
            'account_number': SEPAY_CONFIG['account_number'],
            'account_name': SEPAY_CONFIG['account_name']
        }
    except Exception as e:
        logger.exception("[SEPAY] QR generation error: %s", e)
        return {
            'success': False,
            'error': f'SePay QR generation error: {e}'
        }


# Test function
if __name__ == '__main__':
    # Test tạo payment
    result = create_parking_payment_sepay(session_id=123, amount=5000)
    print("Payment result:", result)


# Re-export for backward compatibility
__all__ = ["create_parking_payment_sepay", "SEPAY_CONFIG", "SEPAY_ENDPOINTS", "sepay", "SePay"]
