"""
SEPAY CONFIGURATION

Cấu hình thanh toán SePay cho hệ thống bãi xe
"""

from typing import Dict, Any

# API Configuration (ĐÚNG THEO TÀI LIỆU)
SEPAY_CONFIG: Dict[str, Any] = {
    # API Configuration (ĐÚNG THEO TÀI LIỆU)
    "api_url": "https://my.sepay.vn/userapi",
    "api_token": "ELFL7N6R3JZHUO1VNOWD1WABUKQXRZMS9XQTBSKG4JPYQIWHL5FEAUI3DVYG4O2X",
    
    # QR Service - VietQR format (api.vietqr.io)
    "qr_url": "https://api.vietqr.io/v2/generate",
    
    # Bank Account Info (CẬP NHẬT THÔNG TIN THẬT)
    "bank_short_name": "VNBA",                    # ✅ FIXED: VietinBank code
    "bank_display_name": "VietinBank",            # Tên hiển thị ngân hàng
    "account_number": "102874512400",             # Số TK thật của bạn
    "account_name": "NGUYEN TUAN ANH",            # Tên thật của bạn
    
    # System Settings
    "webhook_url": "https://parking.epulearn.xyz/api/sepay/webhook",  # ✅ Production URL
    "timeout": 30,  # Timeout cho API calls (seconds)
    "enable_sandbox": True,  # False khi production
    
    # SEVQR PREFIX - YÊU CẦU CỦA SEPAY CHO VIETINBANK
    "content_prefix": "SEVQR"  # Prefix bắt buộc cho VietinBank
}

# Endpoint URLs (ĐÚNG THEO TÀI LIỆU)
SEPAY_ENDPOINTS = {
    "transactions_list": "/transactions/list",      # Lấy danh sách giao dịch
    "transaction_detail": "/transactions/details",  # Chi tiết giao dịch
    "bankaccounts_list": "/bankaccounts/list",      # Danh sách tài khoản
    "bankaccount_detail": "/bankaccounts/details"   # Chi tiết tài khoản
}

# ✅ VIETINBANK BANK CODES:
# - VNBA: ISO 3166 code (used for VietQR)
# - 970405: NAPAS bank code
# - VietinBank: Old format (DEPRECATED)

