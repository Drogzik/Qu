#pragma once
#include <string>

void openApp(const std::string& path);
// Opens URL in a specific browser exe if possible; falls back to default handler.
void openUrlInBrowser(const std::string& browserExe, const std::string& url);
void closeApp(const std::string& exeName);
void toggleMedia();
void volumeUp(int steps);
void volumeDown(int steps);
void volumeMuteToggle();
void scrollPageDown(int steps);
void scrollPageUp(int steps);
void focusAndPressEnter(int tabCount);