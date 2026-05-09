#include "SystemControl.h"
#include <windows.h>
#include <shellapi.h>
#include <iostream>
#include <thread>

void openApp(const std::string& path) {
    ShellExecuteA(NULL, "open", path.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

void openUrlInBrowser(const std::string& browserExe, const std::string& url) {
    // Try explicit browser exe first: "<browserExe>" "<url>"
    HINSTANCE h = ShellExecuteA(NULL, "open", browserExe.c_str(), url.c_str(), NULL, SW_SHOWNORMAL);
    if ((INT_PTR)h <= 32) {
        // Fallback to system default browser handler.
        ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
    }
}

void closeApp(const std::string& exeName) {
    // Используем taskkill через cmd. /F - принудительно, /IM - поиск по имени образа
    std::string command = "taskkill /F /IM " + exeName + " /T > nul 2>&1";
    system(command.c_str());
}

void toggleMedia() {
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0);
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
}

void volumeUp(int steps) {
    if (steps < 1)
        steps = 1;
    for (int i = 0; i < steps; ++i) {
        keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_EXTENDEDKEY, 0);
        keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
    }
}

void volumeDown(int steps) {
    if (steps < 1)
        steps = 1;
    for (int i = 0; i < steps; ++i) {
        keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_EXTENDEDKEY, 0);
        keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
    }
}

void volumeMuteToggle() {
    keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_EXTENDEDKEY, 0);
    keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
}

static void pressVk(WORD vk) {
    keybd_event(static_cast<BYTE>(vk), 0, KEYEVENTF_EXTENDEDKEY, 0);
    keybd_event(static_cast<BYTE>(vk), 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
}

void scrollPageDown(int steps) {
    if (steps < 1) steps = 1;
    for (int i = 0; i < steps; ++i) pressVk(VK_NEXT); // PageDown
}

void scrollPageUp(int steps) {
    if (steps < 1) steps = 1;
    for (int i = 0; i < steps; ++i) pressVk(VK_PRIOR); // PageUp
}

void focusAndPressEnter(int tabCount) {
    if (tabCount < 1) tabCount = 1;
    for (int i = 0; i < tabCount; ++i) pressVk(VK_TAB);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    pressVk(VK_RETURN);
}