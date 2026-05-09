#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "AppWindow.h"
#include "HostBridge.h"

#include <windows.h>
#include <wrl.h>

#include "AI_Engine.h"

#include "WebView2.h"
#include "WebView2EnvironmentOptions.h"

#include <cstdio>
#include <filesystem>

using Microsoft::WRL::ComPtr;

static HWND g_hwnd = nullptr;
static ComPtr<ICoreWebView2Controller> g_controller;
static ComPtr<ICoreWebView2> g_webview;

static constexpr wchar_t kWndClassName[] = L"PomoshnikAppWindow";

const wchar_t* PomoshnikWindowClassName() {
    return kWndClassName;
}

static HWND FindExistingPomoshnikWindow() {
    return FindWindowW(kWndClassName, nullptr);
}

bool PomoshnikActivateRunningInstance() {
    HWND h = FindExistingPomoshnikWindow();
    if (!h) {
        return false;
    }
    if (IsIconic(h)) {
        ShowWindow(h, SW_RESTORE);
    }
    SetForegroundWindow(h);
    return true;
}

static void ShowStartupErrorW(const wchar_t* msg) {
    MessageBoxW(nullptr, msg, L"\u041f\u043e\u043c\u043e\u0448\u043d\u0438\u043a", MB_OK | MB_ICONERROR);
}

namespace {

constexpr UINT_PTR kDeferModelLoadTimerId = 1;
AIEngine* g_aiDeferredModelLoad = nullptr;

} // namespace

static void NotifyWebView2Failure(HRESULT hr) {
    wchar_t buf[512];
    swprintf_s(buf, L"Не удалось загрузить WebView2.\nОбновите Microsoft Edge или поставьте WebView2 Runtime.\nКод ошибки: 0x%08lX\n\nПодробнее: https://developer.microsoft.com/microsoft-edge/webview2/", static_cast<long>(hr));
    ShowStartupErrorW(buf);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case WM_SIZE:
        if (g_controller) {
            RECT bounds;
            GetClientRect(hwnd, &bounds);
            g_controller->put_Bounds(bounds);
        }
        return 0;
    case WM_TIMER:
        if (wparam == kDeferModelLoadTimerId) {
            KillTimer(hwnd, kDeferModelLoadTimerId);
            if (g_aiDeferredModelLoad) {
                g_aiDeferredModelLoad->startBackgroundModelLoad();
                g_aiDeferredModelLoad = nullptr;
            }
            return 0;
        }
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    case WM_DESTROY:
        KillTimer(hwnd, kDeferModelLoadTimerId);
        g_aiDeferredModelLoad = nullptr;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
}

void InitWebView(AIEngine& ai, const std::wstring& userDataDir) {
    auto opts = Microsoft::WRL::Make<CoreWebView2EnvironmentOptions>();
    const HRESULT qr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr,
        userDataDir.c_str(),
        opts.Get(),
        Microsoft::WRL::Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [&, userDataDir](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result)) {
                    NotifyWebView2Failure(result);
                    return result;
                }
                if (!env) {
                    ShowStartupErrorW(L"Среда WebView2 недоступна (env is null).\nОбновите Edge или установите Evergreen WebView2 Runtime.");
                    return E_FAIL;
                }
                env->CreateCoreWebView2Controller(
                    g_hwnd,
                    Microsoft::WRL::Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [&](HRESULT res, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(res)) {
                                NotifyWebView2Failure(res);
                                return res;
                            }
                            if (!controller) {
                                ShowStartupErrorW(L"Не создан контроллер WebView2.\nОбновите Microsoft Edge / WebView2 Runtime.");
                                return E_FAIL;
                            }
                            g_controller = controller;
                            g_controller->get_CoreWebView2(&g_webview);
                            if (!g_webview) {
                                ShowStartupErrorW(L"Не получен объект WebView2.\nОбновите Microsoft Edge / WebView2 Runtime.");
                                return E_FAIL;
                            }
                            // Force dark rendering profile so WebView won't auto-lighten controls/styles.
                            ComPtr<ICoreWebView2_13> webview13;
                            if (SUCCEEDED(g_webview.As(&webview13)) && webview13) {
                                ComPtr<ICoreWebView2Profile> profile;
                                if (SUCCEEDED(webview13->get_Profile(&profile)) && profile) {
                                    profile->put_PreferredColorScheme(COREWEBVIEW2_PREFERRED_COLOR_SCHEME_DARK);
                                }
                            }

                            RECT bounds{};
                            GetClientRect(g_hwnd, &bounds);
                            g_controller->put_Bounds(bounds);

                            EventRegistrationToken token{};
                            g_webview->add_WebMessageReceived(
                                Microsoft::WRL::Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [&, userDataDir](ICoreWebView2* sender,
                                                      ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                        return HandleHostWebMessage(sender, args, ai, userDataDir);
                                    })
                                    .Get(),
                                &token);

                            const std::wstring shellHtml = LoadShellHtmlW();
                            g_webview->NavigateToString(shellHtml.c_str());
                            return S_OK;
                        })
                        .Get());
                return S_OK;
            })
            .Get());

    if (FAILED(qr)) {
        NotifyWebView2Failure(qr);
    }
}

int RunAppWindow(AIEngine& ai, const std::wstring& appDataDir) {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = kWndClassName;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    if (!RegisterClassW(&wc)) {
        const DWORD err = GetLastError();
        // Повторный запуск уже зарегистрировал класс в этом процессе.
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            wchar_t buf[260];
            swprintf_s(buf, L"Не удалось зарегистрировать окно приложения.\nКод ошибки: %lu", static_cast<unsigned long>(err));
            ShowStartupErrorW(buf);
            CoUninitialize();
            return 1;
        }
    }

    g_hwnd = CreateWindowExW(
        0,
        kWndClassName,
        L"\u041f\u043e\u043c\u043e\u0448\u043d\u0438\u043a",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1040,
        800,
        nullptr,
        nullptr,
        wc.hInstance,
        nullptr);

    if (!g_hwnd) {
        const DWORD err = GetLastError();
        wchar_t buf[260];
        swprintf_s(buf, L"Не удалось создать окно.\nКод ошибки: %lu", static_cast<unsigned long>(err));
        ShowStartupErrorW(buf);
        CoUninitialize();
        return 1;
    }

    ShowWindow(g_hwnd, SW_SHOW);
    UpdateWindow(g_hwnd);

    std::filesystem::path ud = std::filesystem::path(appDataDir) / L"webview2";
    std::error_code ec;
    std::filesystem::create_directories(ud, ec);
    InitWebView(ai, ud.wstring());
    // Дать WebView2 и окну прогрузиться до тяжёлого llama/CUDA — первый запуск заметно быстрее.
    g_aiDeferredModelLoad = &ai;
    SetTimer(g_hwnd, kDeferModelLoadTimerId, 900, nullptr);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    g_webview.Reset();
    g_controller.Reset();
    g_hwnd = nullptr;

    UnregisterClassW(kWndClassName, wc.hInstance);
    CoUninitialize();
    return 0;
}
