import sys, subprocess, time, os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style, init

# Fix encoding cho Windows console để hiển thị tiếng Việt
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

init(autoreset=True)

APP_FILE = "parking_ui.py"
DEBOUNCE_TIME = 3.0  # ✅ FIX: Tăng lên 3 giây (từ 1.5)
CAMERA_RELEASE_TIME = 2.0  # ✅ FIX: Thêm thời gian đợi camera

# ✅ Sử dụng Python toàn cục (đã có đầy đủ thư viện)
PYTHON_EXE = sys.executable

class Handler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith(APP_FILE):
            now = time.time()
            # ✅ FIX: Tránh restart liên tục khi file bị lưu nhiều lần
            if now - self.last_modified < DEBOUNCE_TIME:
                print(Fore.YELLOW + f"[Watcher] Bỏ qua thay đổi (debounce: {DEBOUNCE_TIME}s)")
                return
            self.last_modified = now
            print(Fore.YELLOW + f"[Watcher] {APP_FILE} thay đổi, restart app...")
            restart_app()

p = None

def start_app():
    global p
    if p is not None:
        print(Fore.YELLOW + "[Runner] App đang chạy, bỏ qua start_app()")
        return
    print(Fore.GREEN + "[Runner] Bắt đầu chạy app...")
    # ✅ FIX: Unbuffer output để thấy log ngay
    p = subprocess.Popen(
        [PYTHON_EXE, "-u", APP_FILE],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    print(Fore.GREEN + f"[Runner] App started với PID: {p.pid}")

def restart_app():
    """✅ FIXED: Proper cleanup sequence with longer waits"""
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Dừng app cũ...")
        p.terminate()
        
        # ✅ FIX: Tăng timeout lên 5 giây
        try:
            p.wait(timeout=5)
            print(Fore.GREEN + "[Runner] App đã dừng sạch")
        except subprocess.TimeoutExpired:
            print(Fore.RED + "[Runner] App không dừng kịp, kill...")
            p.kill()
            p.wait()
            print(Fore.RED + "[Runner] App đã bị kill")
        
        # ✅ FIX: Đợi camera release hoàn toàn
        print(Fore.YELLOW + f"[Runner] Đợi {CAMERA_RELEASE_TIME}s để camera release...")
        time.sleep(CAMERA_RELEASE_TIME)
    
    print(Fore.GREEN + "[Runner] Chạy lại app...")
    # ✅ FIX: Unbuffer output để thấy log ngay
    p = subprocess.Popen(
        [PYTHON_EXE, "-u", APP_FILE],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    print(Fore.GREEN + f"[Runner] App restarted với PID: {p.pid}")

def stop_app():
    """✅ FIXED: Clean shutdown with KeyboardInterrupt handling"""
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Tắt app...")
        p.terminate()
        try:
            p.wait(timeout=5)  # ✅ FIX: Tăng timeout
            print(Fore.GREEN + "[Runner] App đã tắt")
        except subprocess.TimeoutExpired:
            print(Fore.RED + "[Runner] App không dừng, kill...")
            p.kill()
            p.wait()
            print(Fore.RED + "[Runner] App đã bị kill")
        except KeyboardInterrupt:
            # ✅ Xử lý Ctrl+C trong lúc cleanup
            print(Fore.YELLOW + "[Runner] Ctrl+C trong cleanup, force kill...")
            p.kill()
            try:
                p.wait(timeout=1)
            except Exception:
                pass
            print(Fore.GREEN + "[Runner] App đã tắt (force)")

if __name__ == "__main__":
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "  PARKING APP AUTO-RELOADER")
    print(Fore.CYAN + f"  Debounce: {DEBOUNCE_TIME}s | Camera wait: {CAMERA_RELEASE_TIME}s")
    print(Fore.CYAN + f"  Python: {PYTHON_EXE}")
    print(Fore.CYAN + "=" * 60)
    
    if not os.path.exists(APP_FILE):
        print(Fore.RED + f"[Error] Không tìm thấy file {APP_FILE}")
        sys.exit(1)

    start_app()
    
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print(Fore.GREEN + "[Watcher] Đang theo dõi thay đổi file...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[Watcher] Nhận Ctrl+C, đang dừng...")
        observer.stop()
        stop_app()
        print(Fore.GREEN + "[Watcher] Đã dừng sạch")

    observer.join()