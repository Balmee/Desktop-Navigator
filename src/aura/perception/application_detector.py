import ctypes

import psutil

from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.GetForegroundWindow.restype = wintypes.HWND

user32.GetWindowTextLengthW.argtypes = [
    wintypes.HWND
]

user32.GetWindowTextLengthW.restype = ctypes.c_int

user32.GetWindowTextW.argtypes = [
    wintypes.HWND,
    wintypes.LPWSTR,
    ctypes.c_int
]

user32.GetWindowTextW.restype = ctypes.c_int

user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD)
]


def get_active_application() -> dict:

    hwnd = user32.GetForegroundWindow()

    if not hwnd:
        return {
            "title": None,
            "process": None,
            "pid": None
        }

    length = user32.GetWindowTextLengthW(hwnd)

    buffer = ctypes.create_unicode_buffer(length + 1)

    user32.GetWindowTextW(
        hwnd,
        buffer,
        length + 1
    )

    title = buffer.value

    pid = wintypes.DWORD()

    user32.GetWindowThreadProcessId(
        hwnd,
        ctypes.byref(pid)
    )

    process_name = None

    try:
        process = psutil.Process(pid.value)
        process_name = process.name()

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return {
        "title": title,
        "process": process_name,
        "pid": pid.value
    }