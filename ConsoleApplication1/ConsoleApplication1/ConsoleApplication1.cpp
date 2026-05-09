#include <windows.h>
#include <shellapi.h>

#include <iostream>
#include <string>
#include <filesystem>

#include "SystemControl.h"
#include "AI_Engine.h"
#include "AppWindow.h"

static std::string toUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string out(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), size, nullptr, nullptr);
    return out;
}

static std::wstring toShortPathW(const std::filesystem::path& p) {
    const std::wstring in = p.wstring();
    std::wstring out(MAX_PATH, L'\0');
    DWORD n = GetShortPathNameW(in.c_str(), out.data(), (DWORD)out.size());
    if (n == 0) {
        return in; // fallback
    }
    if (n > out.size()) {
        out.resize(n);
        n = GetShortPathNameW(in.c_str(), out.data(), (DWORD)out.size());
        if (n == 0) {
            return in;
        }
    }
    out.resize(n);
    return out;
}

int main() {
    setlocale(LC_ALL, "Russian");

    wchar_t exePathW[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePathW, MAX_PATH);
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    const std::filesystem::path exeModelsDir = exeDir / "models";
    const std::filesystem::path cwdModelsDir = std::filesystem::current_path() / "models";

    std::filesystem::path modelPath = cwdModelsDir / "llama-3-8b.gguf";
    if (!std::filesystem::exists(modelPath)) {
        modelPath = exeModelsDir / "llama-3-8b.gguf";
    }
    if (!std::filesystem::exists(modelPath)) {
        const std::filesystem::path dirs[] = { cwdModelsDir, exeModelsDir };
        for (const auto& dir : dirs) {
            if (!std::filesystem::exists(dir)) {
                continue;
            }
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                    modelPath = entry.path();
                    break;
                }
            }
            if (std::filesystem::exists(modelPath)) {
                break;
            }
        }
    }

    // If still not found, walk up from exeDir to locate <some parent>/models/*.gguf
    if (!std::filesystem::exists(modelPath)) {
        std::filesystem::path cur = exeDir;
        for (int up = 0; up < 6; ++up) {
            const std::filesystem::path candidateModels = cur / "models";
            if (std::filesystem::exists(candidateModels)) {
                for (const auto& entry : std::filesystem::directory_iterator(candidateModels)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                        modelPath = entry.path();
                        break;
                    }
                }
                if (std::filesystem::exists(modelPath)) {
                    break;
                }
            }
            if (!cur.has_parent_path()) {
                break;
            }
            cur = cur.parent_path();
        }
    }

    const std::wstring modelShortW = toShortPathW(modelPath);
    const std::string modelShortUtf8 = toUtf8(modelShortW);
    std::cout << "Model file: " << modelShortUtf8 << std::endl;

    AIEngine ai(modelShortUtf8);

    int argc = 0;
    LPWSTR* argvW = CommandLineToArgvW(GetCommandLineW(), &argc);
    bool consoleMode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::wstring(argvW[i]) == L"--console") {
            consoleMode = true;
        }
    }
    if (argvW) LocalFree(argvW);

    if (!consoleMode) {
        const std::filesystem::path appDataDir = exeDir / "pomoshnik_data";
        std::error_code ec;
        std::filesystem::create_directories(appDataDir, ec);
        return RunAppWindow(ai, appDataDir.wstring());
    }

    std::cout << "Pomoshnik s AI zapushen." << std::endl;

    std::string userInput;
    while (true) {
        std::cout << "Ty: ";
        std::getline(std::cin, userInput); // Используем getline для длинных фраз

        if (userInput == "vse") break;

        std::string aiResponse = ai.generateResponse(userInput);

        if (aiResponse == "COMMAND_KILL_NOTEPAD") {
            closeApp("notepad.exe");
            std::cout << "AI: Zakryvayu bloknot по твоей просьбе." << std::endl;
        }
        else if (aiResponse == "COMMAND_OPEN_NOTEPAD") {
            openApp("notepad.exe");
            std::cout << "AI: Otkryvayu bloknot." << std::endl;
        }
        else {
            std::cout << "AI: " << aiResponse << std::endl;
        }
    }
    return 0;
}