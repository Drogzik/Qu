#include <windows.h>
#include <shellapi.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cwctype>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <shlobj.h>

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

static std::filesystem::path FirstGgufInDirHint(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) {
        return {};
    }
    std::filesystem::path best;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        std::wstring ext = entry.path().extension().wstring();
        for (auto& ch : ext) {
            ch = static_cast<wchar_t>(std::towlower(static_cast<wint_t>(ch)));
        }
        if (ext != L".gguf") {
            continue;
        }
        const auto name = entry.path().filename().wstring();
        if (name.find(L"llama") != std::wstring::npos) {
            return entry.path();
        }
        if (best.empty()) {
            best = entry.path();
        }
    }
    return best;
}

// Те же правила поиска, что и в AIEngine::refreshModelPathFromDiskLocked (подсказка при старте).
static std::filesystem::path ResolveModelPath() {
    wchar_t exePathW[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePathW, MAX_PATH);
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    const std::filesystem::path exeModelsDir = exeDir / L"models";
    std::error_code ec;
    const std::filesystem::path cwdModelsDir = std::filesystem::current_path(ec) / L"models";

    std::filesystem::path found = FirstGgufInDirHint(cwdModelsDir);
    if (!found.empty()) {
        return found;
    }
    found = FirstGgufInDirHint(exeModelsDir);
    if (!found.empty()) {
        return found;
    }

    wchar_t docs[MAX_PATH] = {};
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_MYDOCUMENTS, nullptr, SHGFP_TYPE_CURRENT, docs))) {
        found = FirstGgufInDirHint(std::filesystem::path(docs) / L"Pomoshnik" / L"models");
        if (!found.empty()) {
            return found;
        }
    }

    wchar_t profileDir[32768] = {};
    const DWORD np = GetEnvironmentVariableW(L"USERPROFILE", profileDir, static_cast<DWORD>(std::size(profileDir)));
    if (np > 0 && np < std::size(profileDir)) {
        const std::filesystem::path prof(profileDir);
        found = FirstGgufInDirHint(prof / L"Downloads" / L"Pomoshnik" / L"models");
        if (!found.empty()) {
            return found;
        }
        found = FirstGgufInDirHint(prof / L"Загрузки" / L"Pomoshnik" / L"models");
        if (!found.empty()) {
            return found;
        }
    }

    wchar_t* ev = nullptr;
    size_t evLen = 0;
    if (_wdupenv_s(&ev, &evLen, L"POMOSHNIK_MODEL_DIR") == 0 && ev && ev[0]) {
        found = FirstGgufInDirHint(std::filesystem::path(ev));
        std::free(ev);
        if (!found.empty()) {
            return found;
        }
    } else {
        std::free(ev);
    }

    std::filesystem::path cur = exeDir;
    for (int up = 0; up < 6; ++up) {
        found = FirstGgufInDirHint(cur / L"models");
        if (!found.empty()) {
            return found;
        }
        if (!cur.has_parent_path()) {
            break;
        }
        cur = cur.parent_path();
    }

    return {};
}

static bool DirHasLargeGgufW(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        std::wstring ext = entry.path().extension().wstring();
        for (auto& ch : ext) {
            ch = static_cast<wchar_t>(std::towlower(static_cast<wint_t>(ch)));
        }
        if (ext != L".gguf") {
            continue;
        }
        const auto sz = entry.file_size(ec);
        if (ec) {
            continue;
        }
        if (sz > 50ull * 1024 * 1024) {
            return true;
        }
    }
    return false;
}

// Если рядом с exe лежит download_model.ps1 и в models\ нет .gguf — скачиваем (инсталлятор мог не догрузить).
static void TryRunBundledModelDownloader(const std::filesystem::path& exeDir) {
    std::error_code ec;
    const auto modelsDir = exeDir / L"models";
    if (DirHasLargeGgufW(modelsDir)) {
        return;
    }
    const auto script = exeDir / L"download_model.ps1";
    if (!std::filesystem::exists(script, ec) || ec) {
        return;
    }
    const std::wstring ps =
        L"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
    const std::wstring cmdLine =
        L"\"" + ps + L"\" -NoProfile -ExecutionPolicy Bypass -File \"" + script.wstring() + L"\" \"" +
        exeDir.wstring() + L"\"";

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmdLine.begin(), cmdLine.end());
    buf.push_back(L'\0');

    const std::wstring workDir = exeDir.wstring();
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                        workDir.c_str(), &si, &pi)) {
        return;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
}

// GUI: данные всегда в %LOCALAPPDATA%\Pomoshnik (не рядом с exe).
// Свой путь: переменная окружения POMOSHNIK_APPDATA (полный путь к каталогу).
static std::filesystem::path ResolveGuiAppDataDir(const std::filesystem::path& exeDir) {
    wchar_t* ev = nullptr;
    size_t evLen = 0;
    const errno_t evErr = _wdupenv_s(&ev, &evLen, L"POMOSHNIK_APPDATA");
    if (evErr == 0 && ev && ev[0]) {
        std::filesystem::path custom(ev);
        std::free(ev);
        ev = nullptr;
        std::error_code ec;
        std::filesystem::create_directories(custom, ec);
        return custom;
    }
    std::free(ev);

    wchar_t localBuf[MAX_PATH] = {};
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA, nullptr, SHGFP_TYPE_CURRENT, localBuf))) {
        std::filesystem::path dir = std::filesystem::path(localBuf) / L"Pomoshnik";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);

        const std::filesystem::path legacyState = exeDir / L"pomoshnik_data" / L"app_state.json";
        const std::filesystem::path newState = dir / L"app_state.json";
        if (!std::filesystem::exists(newState, ec) && std::filesystem::exists(legacyState, ec)) {
            std::filesystem::copy_file(legacyState, newState, std::filesystem::copy_options::overwrite_existing, ec);
        }
        return dir;
    }

    std::filesystem::path fallback = exeDir / L"pomoshnik_data";
    std::error_code ec;
    std::filesystem::create_directories(fallback, ec);
    return fallback;
}

static int RunConsoleMode(AIEngine& ai) {
    setlocale(LC_ALL, "Russian");
    std::cout << "Pomoshnik s AI zapushen." << std::endl;

    std::string userInput;
    while (true) {
        std::cout << "Ty: ";
        std::getline(std::cin, userInput);

        if (userInput == "vse") break;

        std::string aiResponse = ai.generateResponse(userInput);

        if (aiResponse == "COMMAND_KILL_NOTEPAD") {
            closeApp("notepad.exe");
            std::cout << "AI: Zakryvayu bloknot po tvoej prosjbe." << std::endl;
        } else if (aiResponse == "COMMAND_OPEN_NOTEPAD") {
            openApp("notepad.exe");
            std::cout << "AI: Otkryvayu bloknot." << std::endl;
        } else {
            std::cout << "AI: " << aiResponse << std::endl;
        }
    }
    return 0;
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int argc = 0;
    LPWSTR* argvW = CommandLineToArgvW(GetCommandLineW(), &argc);
    bool consoleMode = false;
    for (int i = 1; i < argc; ++i) {
        if (argvW && std::wstring(argvW[i]) == L"--console") {
            consoleMode = true;
        }
    }
    if (argvW) {
        LocalFree(argvW);
    }

    wchar_t exePathW[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePathW, MAX_PATH);
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    {
        std::error_code ec;
        std::filesystem::create_directories(exeDir / L"models", ec);
    }

    if (consoleMode) {
        TryRunBundledModelDownloader(exeDir);
        const std::filesystem::path modelPath = ResolveModelPath();
        const std::string modelShortUtf8 =
            modelPath.empty() ? std::string{} : toUtf8(modelPath.wstring());
        if (!AllocConsole()) {
            return 1;
        }
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        FILE* f = nullptr;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONIN$", "r", stdin);
        freopen_s(&f, "CONOUT$", "w", stderr);
        std::cout << "Model file: " << modelShortUtf8 << std::endl;
        AIEngine ai(modelShortUtf8);
        std::cout << "Загрузка модели..." << std::endl;
        ai.preloadModel();
        return RunConsoleMode(ai);
    }

    static constexpr wchar_t kDesktopSingleInstanceMutexName[] = L"Local\\PomoshnikDesktopAppSingleInstanceV1";
    HANDLE hSingleInstanceMutex = nullptr;
    SetLastError(0);
    hSingleInstanceMutex = CreateMutexW(nullptr, TRUE, kDesktopSingleInstanceMutexName);
    if (!hSingleInstanceMutex) {
        // Нет блокировки — приложение может запуститься вторым процессом.
    } else {
        const DWORD mtxErr = GetLastError();
        if (mtxErr == ERROR_ALREADY_EXISTS) {
            PomoshnikActivateRunningInstance();
            CloseHandle(hSingleInstanceMutex);
            hSingleInstanceMutex = nullptr;
            return 0;
        }
    }

    TryRunBundledModelDownloader(exeDir);
    const std::filesystem::path modelPath = ResolveModelPath();
    const std::string modelShortUtf8 =
        modelPath.empty() ? std::string{} : toUtf8(modelPath.wstring());

    // Без .gguf окно открывается: закладки, команды, настройки работают; полноценные ответы ИИ — после .gguf в models\.
    AIEngine ai(modelShortUtf8);

    const std::filesystem::path appDataDir = ResolveGuiAppDataDir(exeDir);
    const int windowExitCode = RunAppWindow(ai, appDataDir.wstring());
    if (hSingleInstanceMutex) {
        ReleaseMutex(hSingleInstanceMutex);
        CloseHandle(hSingleInstanceMutex);
    }
    return windowExitCode;
}
