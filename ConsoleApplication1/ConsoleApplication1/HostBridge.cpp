#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "HostBridge.h"

#include <windows.h>

#include "AI_Engine.h"
#include "SystemControl.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static std::string g_preferredBrowserExe = "msedge.exe";
static std::string g_preferredTemperament = "friendly";

static bool FileExistsA(const std::string& path) {
    if (path.empty())
        return false;
    DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

static std::string ResolveBrowserExe(const std::string& browserKey) {
    auto appPath = [](const char* exe) -> std::string {
        // Try App Paths in HKCU then HKLM.
        char buf[1024];
        DWORD sz = sizeof(buf);
        const std::string sub = std::string("Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\") + exe;
        if (RegGetValueA(HKEY_CURRENT_USER, sub.c_str(), nullptr, RRF_RT_REG_SZ, nullptr, buf, &sz) == ERROR_SUCCESS) {
            return std::string(buf);
        }
        sz = sizeof(buf);
        if (RegGetValueA(HKEY_LOCAL_MACHINE, sub.c_str(), nullptr, RRF_RT_REG_SZ, nullptr, buf, &sz) == ERROR_SUCCESS) {
            return std::string(buf);
        }
        return std::string(exe);
    };

    auto commonPath = [](const char* p1, const char* p2) -> std::string {
        if (p1 && *p1) {
            DWORD a = GetFileAttributesA(p1);
            if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY))
                return std::string(p1);
        }
        if (p2 && *p2) {
            DWORD a = GetFileAttributesA(p2);
            if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY))
                return std::string(p2);
        }
        return {};
    };

    if (browserKey == "firefox") {
        std::string p = appPath("firefox.exe");
        if (FileExistsA(p))
            return p;
        p = commonPath("C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                       "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe");
        return p.empty() ? std::string("firefox.exe") : p;
    }
    if (browserKey == "chrome") {
        std::string p = appPath("chrome.exe");
        if (FileExistsA(p))
            return p;
        p = commonPath("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                       "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe");
        return p.empty() ? std::string("chrome.exe") : p;
    }
    if (browserKey == "yandex") {
        std::string p = appPath("browser.exe");
        if (FileExistsA(p))
            return p;
        p = commonPath("C:\\Program Files\\Yandex\\YandexBrowser\\Application\\browser.exe",
                       "C:\\Program Files (x86)\\Yandex\\YandexBrowser\\Application\\browser.exe");
        return p.empty() ? std::string("browser.exe") : p;
    }
    if (browserKey == "opera") {
        std::string p = appPath("opera.exe");
        if (FileExistsA(p))
            return p;
        p = commonPath("C:\\Program Files\\Opera\\launcher.exe",
                       "C:\\Program Files (x86)\\Opera\\launcher.exe");
        // Opera чаще launcher.exe
        return p.empty() ? std::string("opera.exe") : p;
    }

    // Edge
    {
        std::string p = appPath("msedge.exe");
        if (FileExistsA(p))
            return p;
        p = commonPath("C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
                       "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe");
        return p.empty() ? std::string("msedge.exe") : p;
    }
}

static std::string SanitizeUtf8Lossy(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size();) {
        const uint8_t c = static_cast<uint8_t>(in[i]);
        if (c < 0x80u) {
            out.push_back(static_cast<char>(c));
            ++i;
            continue;
        }
        size_t need = 0;
        if ((c >> 5u) == 0x6u)
            need = 2;
        else if ((c >> 4u) == 0xEu)
            need = 3;
        else if ((c >> 3u) == 0x1Eu)
            need = 4;
        else {
            ++i;
            continue;
        }

        if (i + need > in.size()) {
            ++i;
            continue;
        }
        bool ok = true;
        for (size_t j = 1; j < need; ++j) {
            if ((static_cast<uint8_t>(in[i + j]) & 0xC0u) != 0x80u) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            ++i;
            continue;
        }
        out.append(in.data() + i, need);
        i += need;
    }
    return out;
}

static std::wstring Utf8ToWideSafe(const std::string& utf8) {
    const std::string s = SanitizeUtf8Lossy(utf8);
    if (s.empty()) {
        return {};
    }
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (len <= 0) {
        return {};
    }
    std::wstring w(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), len);
    return w;
}

static int HexDigit(char h) {
    if (h >= '0' && h <= '9')
        return h - '0';
    if (h >= 'a' && h <= 'f')
        return 10 + h - 'a';
    if (h >= 'A' && h <= 'F')
        return 10 + h - 'A';
    return -1;
}

static void AppendUtf8FromCodepoint(uint32_t cp, std::string& out) {
    if (cp <= 0x7Fu) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7FFu) {
        out += static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu));
        out += static_cast<char>(0x80u | (cp & 0x3Fu));
    } else if (cp <= 0xFFFFu) {
        out += static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu));
        out += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
        out += static_cast<char>(0x80u | (cp & 0x3Fu));
    }
}

static std::string ExtractJsonStringUtf8(const std::string& json, const std::string& field) {
    const std::string key = "\"" + field + "\":\"";
    size_t p = json.find(key);
    if (p == std::string::npos) {
        return {};
    }
    p += key.size();
    std::string out;
    while (p < json.size()) {
        const char c = json[p];
        if (c == '\\' && p + 1 < json.size()) {
            const char n = json[p + 1];
            if (n == 'n') {
                out += '\n';
                p += 2;
                continue;
            }
            if (n == 'r') {
                out += '\r';
                p += 2;
                continue;
            }
            if (n == '"' || n == '\\') {
                out += n;
                p += 2;
                continue;
            }
            if (n == 'u' && p + 5 < json.size()) {
                uint32_t cp = 0;
                bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    const int v = HexDigit(json[p + 2 + k]);
                    if (v < 0) {
                        ok = false;
                        break;
                    }
                    cp = (cp << 4) | static_cast<uint32_t>(v);
                }
                if (ok && cp != 0) {
                    AppendUtf8FromCodepoint(cp, out);
                }
                p += 6;
                continue;
            }
            out += n;
            p += 2;
            continue;
        }
        if (c == '"') {
            break;
        }
        out += c;
        ++p;
    }
    return out;
}

static std::string ReadFileUtf8OrEmpty(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in)
        return {};
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

static bool WriteFileUtf8(const std::filesystem::path& p, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return static_cast<bool>(out);
}

static std::string ExtractJsonObjectByFieldUtf8(const std::string& json, const std::string& field) {
    const std::string key = "\"" + field + "\":";
    size_t p = json.find(key);
    if (p == std::string::npos) {
        return {};
    }
    p += key.size();
    while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) {
        ++p;
    }
    if (p >= json.size() || json[p] != '{') {
        return {};
    }
    size_t start = p;
    int depth = 0;
    bool inString = false;
    bool escape = false;
    for (; p < json.size(); ++p) {
        const char c = json[p];
        if (inString) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }
        if (c == '"') {
            inString = true;
            continue;
        }
        if (c == '{') {
            ++depth;
            continue;
        }
        if (c == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(start, p - start + 1);
            }
        }
    }
    return {};
}

static std::string ToLowerAscii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static std::string UrlEncode(const std::string& s) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                          c == '.' || c == '~';
        if (safe) {
            out.push_back(static_cast<char>(c));
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0xF]);
            out.push_back(kHex[c & 0xF]);
        }
    }
    return out;
}

static int ExtractMinuteNumber(const std::string& text) {
    // Non-throwing parser: looks for "<digits> мин/min/minute".
    const auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
    for (size_t i = 0; i < text.size(); ++i) {
        if (!is_digit(static_cast<unsigned char>(text[i])))
            continue;
        size_t j = i;
        int v = 0;
        int digits = 0;
        while (j < text.size() && is_digit(static_cast<unsigned char>(text[j])) && digits < 3) {
            v = v * 10 + (text[j] - '0');
            ++j;
            ++digits;
        }
        if (digits == 0)
            continue;
        size_t k = j;
        while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
        auto starts = [&](const char* w) {
            const size_t n = std::strlen(w);
            return k + n <= text.size() && text.compare(k, n, w) == 0;
        };
        // Match ASCII "min"/"minute" and Cyrillic "мин" (UTF-8 bytes).
        if (starts("min") || starts("minute") || text.find("\xD0\xBC\xD0\xB8\xD0\xBD", k) == k) {
            return v;
        }
    }
    return -1;
}

static std::string TrimAscii(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static void ReplaceAllInPlace(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string ExtractAnimeTitleHint(std::string text) {
    ReplaceAllInPlace(text, "запусти", " ");
    ReplaceAllInPlace(text, "открой", " ");
    ReplaceAllInPlace(text, "включи", " ");
    ReplaceAllInPlace(text, "аниме", " ");
    ReplaceAllInPlace(text, "anime", " ");
    return TrimAscii(text);
}

static int ExtractVideoIndex(const std::string& text) {
    // Extracts N from patterns like "2 видео", "1 видео", "первое видео".
    if (text.find("первое видео") != std::string::npos)
        return 1;
    // Scan for digits followed by "видео"
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < '0' || c > '9')
            continue;
        int v = 0;
        size_t j = i;
        int digits = 0;
        while (j < text.size()) {
            unsigned char d = static_cast<unsigned char>(text[j]);
            if (d < '0' || d > '9')
                break;
            v = v * 10 + (text[j] - '0');
            ++j;
            ++digits;
            if (digits >= 3)
                break;
        }
        if (v <= 0)
            continue;
        // skip spaces
        size_t k = j;
        while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k])))
            ++k;
        // UTF-8 "видео" begins with D0 B2...
        if (k + 2 < text.size() && static_cast<unsigned char>(text[k]) == 0xD0u) {
            // quick substring check
            if (text.find("\xD0\xB2\xD0\xB8\xD0\xB4\xD0\xB5\xD0\xBE", k) == k) {
                return v;
            }
        }
    }
    return -1;
}

static void ParseTurnsArrayUtf8(const std::string& json, std::vector<std::pair<std::string, std::string>>& out) {
    out.clear();
    const std::string key = "\"turns\":[";
    size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return;
    }
    pos += key.size();
    while (pos < json.size()) {
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        if (pos >= json.size()) {
            return;
        }
        if (json[pos] == ']') {
            return;
        }
        if (json[pos] != '{') {
            return;
        }
        size_t depth = 0;
        size_t i = pos;
        for (; i < json.size(); ++i) {
            if (json[i] == '{')
                ++depth;
            else if (json[i] == '}') {
                --depth;
                if (depth == 0) {
                    ++i;
                    break;
                }
            }
        }
        const std::string obj = json.substr(pos, i - pos);
        pos = i;
        const std::string role = ExtractJsonStringUtf8(obj, "role");
        const std::string text = ExtractJsonStringUtf8(obj, "text");
        if ((role == "user" || role == "assistant") && !text.empty()) {
            out.emplace_back(role, text);
        }
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
            ++pos;
        }
    }
}

static std::string JsonEscapeUtf8(const std::string& in) {
    std::string s = SanitizeUtf8Lossy(in);
    std::string escaped;
    for (char c : s) {
        switch (c) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += c;
            break;
        }
    }
    return escaped;
}

static std::wstring MakeJsonReplyW(const std::string& replyUtf8) {
    const std::string json =
        std::string("{\"type\":\"reply\",\"text\":\"") + JsonEscapeUtf8(replyUtf8) + "\"}";
    return Utf8ToWideSafe(json);
}

static std::wstring MakeJsonTypingW(bool on) {
    return on ? Utf8ToWideSafe("{\"type\":\"typing\",\"value\":true}")
              : Utf8ToWideSafe("{\"type\":\"typing\",\"value\":false}");
}

static bool LooksLikeJsonObject(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    return i < s.size() && s[i] == '{';
}

static std::filesystem::path GetExecutableDirW() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (!n || n >= MAX_PATH)
        return {};
    std::filesystem::path p(buf);
    return p.parent_path();
}

std::wstring LoadShellHtmlW() {
    const auto path = GetExecutableDirW() / L"ui" / L"app_shell.html";
    std::string utf8 = ReadFileUtf8OrEmpty(path);
    if (!utf8.empty())
        return Utf8ToWideSafe(utf8);
    return Utf8ToWideSafe(
        "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"color-scheme\" content=\"dark\"></head>"
        "<body style=\"background:#0b0f1d;color:#ecf0ff;font-family:Segoe UI,sans-serif;padding:28px;\">"
        "<h2 style=\"margin:0 0 12px\">Missing ui/app_shell.html</h2>"
        "<p style=\"opacity:.85;margin:0\">Place <code>app_shell.html</code> under <code>ui\\</code> next to the "
        "executable (same folder as Pomoshnik.exe, or next to the built .exe when running from Visual Studio).</p>"
        "</body></html>");
}

HRESULT HandleHostWebMessage(ICoreWebView2* sender, ICoreWebView2WebMessageReceivedEventArgs* args, AIEngine& ai,
                             const std::wstring& userDataDir)
{
    try {
    LPWSTR msgW = nullptr;
    args->TryGetWebMessageAsString(&msgW);
    std::wstring payload = msgW ? msgW : L"";
    if (msgW)
        CoTaskMemFree(msgW);

    std::string userText;
    if (!payload.empty()) {
        int needed = WideCharToMultiByte(
            CP_UTF8, 0, payload.c_str(), (int)payload.size(), nullptr, 0, nullptr, nullptr);
        userText.resize(static_cast<size_t>(needed));
        WideCharToMultiByte(
            CP_UTF8,
            0,
            payload.c_str(),
            (int)payload.size(),
            userText.data(),
            needed,
            nullptr,
            nullptr);
    }

    auto postW = [sender](const std::wstring& w) { sender->PostWebMessageAsString(w.c_str()); };

    if (LooksLikeJsonObject(userText)) {
        const std::string type = ExtractJsonStringUtf8(userText, "type");
        if (type == "reset") {
            ai.resetConversation();
            postW(Utf8ToWideSafe(R"({"type":"ack","action":"reset"})"));
            return S_OK;
        }
        if (type == "syncSession") {
            std::vector<std::pair<std::string, std::string>> turns;
            ParseTurnsArrayUtf8(userText, turns);
            ai.loadHistoryTurns(turns);
            postW(Utf8ToWideSafe(R"({"type":"ack","action":"syncSession"})"));
            return S_OK;
        }
        if (type == "settings") {
            const std::string temperament = ExtractJsonStringUtf8(userText, "temperament");
            const std::string browser = ExtractJsonStringUtf8(userText, "browser");
            if (!temperament.empty()) {
                g_preferredTemperament = temperament;
            }
            if (!browser.empty()) {
                g_preferredBrowserExe = ResolveBrowserExe(browser);
            }
            postW(Utf8ToWideSafe(R"({"type":"ack","action":"settings"})"));
            return S_OK;
        }
        if (type == "storeGet") {
            const std::filesystem::path p = std::filesystem::path(userDataDir) / "app_state.json";
            std::string v = ReadFileUtf8OrEmpty(p);
            if (v.empty() || !LooksLikeJsonObject(v)) {
                v = "{}";
            }
            const std::wstring msg = Utf8ToWideSafe(std::string("{\"type\":\"store\",\"value\":") + v + "}");
            postW(msg);
            return S_OK;
        }
        if (type == "storeSet") {
            // payload format: {"type":"storeSet","value":{...}}
            // Parse value object robustly (respect nested braces/strings).
            const std::string v = ExtractJsonObjectByFieldUtf8(userText, "value");
            if (!v.empty() && LooksLikeJsonObject(v)) {
                const std::filesystem::path p = std::filesystem::path(userDataDir) / "app_state.json";
                const bool ok = WriteFileUtf8(p, v);
                postW(Utf8ToWideSafe(ok
                                         ? R"({"type":"ack","action":"storeSet","ok":true})"
                                         : R"({"type":"ack","action":"storeSet","ok":false})"));
                return S_OK;
            }
            postW(Utf8ToWideSafe(R"({"type":"ack","action":"storeSet","ok":false})"));
            return S_OK;
        }
        if (type == "chat") {
            const std::string text = ExtractJsonStringUtf8(userText, "text");
            const std::string tone = ExtractJsonStringUtf8(userText, "tone");
            const std::string effectiveTone = tone.empty() ? g_preferredTemperament : tone;
            const std::string& cmdSrc = text.empty() ? userText : text;
            const std::string cmdLower = ToLowerAscii(cmdSrc);

            const bool wantsOpen =
                (cmdSrc.find("otkroi") != std::string::npos) ||
                (cmdSrc.find("open") != std::string::npos) ||
                (cmdSrc.find("start") != std::string::npos) ||
                (cmdSrc.find("launch") != std::string::npos) ||
                (cmdSrc.find("включи") != std::string::npos) ||
                (cmdSrc.find("Включи") != std::string::npos) ||
                (cmdSrc.find("запусти") != std::string::npos) ||
                (cmdSrc.find("Запусти") != std::string::npos) ||
                (cmdSrc.find("открой") != std::string::npos) ||
                (cmdSrc.find("Открой") != std::string::npos);
            const bool wantsClose =
                (cmdSrc.find("zakroi") != std::string::npos) ||
                (cmdSrc.find("kill") != std::string::npos) ||
                (cmdSrc.find("закрой") != std::string::npos) ||
                (cmdSrc.find("Закрой") != std::string::npos);
            const bool mentionsNotepad =
                (cmdSrc.find("notepad") != std::string::npos) ||
                (cmdSrc.find("Notepad") != std::string::npos) ||
                (cmdSrc.find("блокнот") != std::string::npos) ||
                (cmdSrc.find("Блокнот") != std::string::npos);
            const bool mentionsBrowser =
                (cmdSrc.find("browser") != std::string::npos) ||
                (cmdSrc.find("Browser") != std::string::npos) ||
                (cmdSrc.find("браузер") != std::string::npos) ||
                (cmdSrc.find("Браузер") != std::string::npos);
            const bool mentionsAnime =
                (cmdSrc.find("anime") != std::string::npos) ||
                (cmdSrc.find("Anime") != std::string::npos) ||
                (cmdSrc.find("аниме") != std::string::npos) ||
                (cmdSrc.find("Аниме") != std::string::npos);
            const bool mentionsYoutube = (cmdLower.find("youtube") != std::string::npos) ||
                                        (cmdSrc.find("YouTube") != std::string::npos) ||
                                        (cmdSrc.find("ютуб") != std::string::npos) ||
                                        (cmdSrc.find("Ютуб") != std::string::npos);
            const bool mentionsSpotify = (cmdLower.find("spotify") != std::string::npos) ||
                                        (cmdSrc.find("спотиф") != std::string::npos);
            const bool mentionsPlaylist =
                (cmdLower.find("playlist") != std::string::npos) ||
                (cmdSrc.find("плейлист") != std::string::npos);
            const bool mentionsMusic = (cmdLower.find("music") != std::string::npos) ||
                                      (cmdSrc.find("музык") != std::string::npos);
            const bool wantsLouder = (cmdSrc.find("громче") != std::string::npos);
            const bool wantsQuieter = (cmdSrc.find("тише") != std::string::npos);
            const bool wantsMute = (cmdSrc.find("выключи звук") != std::string::npos) ||
                                   (cmdLower.find("mute") != std::string::npos);
            const bool wantsScrollDown = (cmdSrc.find("листай вниз") != std::string::npos) ||
                                         (cmdSrc.find("пролистай вниз") != std::string::npos) ||
                                         (cmdLower.find("scroll down") != std::string::npos);
            const bool wantsScrollUp = (cmdSrc.find("листай вверх") != std::string::npos) ||
                                       (cmdSrc.find("пролистай вверх") != std::string::npos) ||
                                       (cmdLower.find("scroll up") != std::string::npos);
            const bool wantsLike = (cmdSrc.find("поставь лайк") != std::string::npos) ||
                                   (cmdLower.find("like video") != std::string::npos);
            const int minuteMark = ExtractMinuteNumber(cmdLower);
            const std::string animeHint = ExtractAnimeTitleHint(cmdSrc);
            const bool wantsSearch = (cmdLower.find("search") != std::string::npos) ||
                                     (cmdSrc.find("поиск") != std::string::npos) ||
                                     (cmdSrc.find("найди") != std::string::npos);
            const bool wantsFirstVideo = (cmdSrc.find("1 видео") != std::string::npos) ||
                                         (cmdSrc.find("первое видео") != std::string::npos) ||
                                         (cmdLower.find("first video") != std::string::npos);
            const int wantsVideoIndex = ExtractVideoIndex(cmdSrc);

            postW(MakeJsonTypingW(true));

            std::string replyUtf8;
            if (wantsOpen && mentionsNotepad) {
                openApp("notepad.exe");
                replyUtf8 = "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                           "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82. "
                           "\xd0\x95\xd1\x81\xd0\xbb\xd0\xb8 \xd0\xbd\xd0\xb0\xd0\xb4\xd0\xbe "
                           "\xe2\x80\x94 \xd0\xbf\xd0\xbe\xd0\xbc\xd0\xbe\xd0\xb3\xd1\x83 "
                           "\xd1\x81 \xd0\xb2\xd0\xb2\xd0\xbe\xd0\xb4\xd0\xbe\xd0\xbc.";
            } else if (wantsOpen && mentionsAnime) {
                if (!animeHint.empty()) {
                    // Open first matching Jut-su page directly (I'm Feeling Lucky).
                    openUrlInBrowser(g_preferredBrowserExe,
                                     std::string("https://www.google.com/search?btnI=1&q=site%3Ajut-su.net+") +
                                         UrlEncode(animeHint));
                    replyUtf8 = "Открыла аниме по названию.";
                } else {
                    openUrlInBrowser(g_preferredBrowserExe, "https://jut-su.net/");
                    replyUtf8 = "Открыла сайт для просмотра аниме.";
                }
            } else if (wantsLouder) {
                volumeUp(6);
                replyUtf8 = "Сделала громче.";
            } else if (wantsQuieter) {
                volumeDown(6);
                replyUtf8 = "Сделала тише.";
            } else if (wantsMute) {
                volumeMuteToggle();
                replyUtf8 = "Переключила mute.";
            } else if (wantsScrollDown) {
                scrollPageDown(2);
                replyUtf8 = "Листнула вниз.";
            } else if (wantsScrollUp) {
                scrollPageUp(2);
                replyUtf8 = "Листнула вверх.";
            } else if (wantsLike) {
                // Best effort: focus page and activate likely actionable control.
                focusAndPressEnter(10);
                replyUtf8 = "Пробую поставить лайк (best effort).";
            } else if ((wantsOpen || mentionsMusic) && mentionsSpotify) {
                openUrlInBrowser(g_preferredBrowserExe, "https://open.spotify.com/");
                replyUtf8 = "Открыла Spotify.";
            } else if ((wantsOpen || mentionsMusic) && mentionsYoutube && mentionsPlaylist) {
                openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/feed/playlists");
                replyUtf8 = "Открыла YouTube плейлисты.";
            } else if (wantsOpen && mentionsYoutube && !mentionsMusic && !mentionsPlaylist && !wantsSearch &&
                       minuteMark <= 0) {
                openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
                replyUtf8 = "Открыла YouTube.";
            } else if ((wantsOpen || wantsSearch) && (wantsFirstVideo || wantsVideoIndex > 0)) {
                std::string q = cmdSrc;
                ReplaceAllInPlace(q, "открой", " ");
                ReplaceAllInPlace(q, "запусти", " ");
                ReplaceAllInPlace(q, "включи", " ");
                ReplaceAllInPlace(q, "ютуб", " ");
                ReplaceAllInPlace(q, "youtube", " ");
                ReplaceAllInPlace(q, "1 видео", " ");
                ReplaceAllInPlace(q, "первое видео", " ");
                ReplaceAllInPlace(q, "видео", " ");
                q = TrimAscii(q);
                if (q.empty()) {
                    openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
                    replyUtf8 = "Открыла YouTube. Напиши: найди <запрос> и открой 1 видео.";
                } else {
                    const int n = (wantsVideoIndex > 0) ? wantsVideoIndex : 1;
                    const int start = (n > 1) ? (n - 1) : 0;
                    // "I'm Feeling Lucky" for Nth watch page directly (num=1, start=N-1).
                    openUrlInBrowser(g_preferredBrowserExe,
                                     std::string("https://www.google.com/search?btnI=1&num=1&start=") +
                                         std::to_string(start) + "&q=site%3Ayoutube.com%2Fwatch+" +
                                         UrlEncode(q));
                    replyUtf8 = "Открываю видео по запросу.";
                }
            } else if (wantsOpen && mentionsYoutube && !wantsSearch && minuteMark <= 0) {
                openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
                replyUtf8 = "Открыла YouTube.";
            } else if ((wantsOpen || mentionsMusic || wantsSearch) && mentionsYoutube) {
                std::string q = UrlEncode(cmdSrc);
                std::string url = "https://www.youtube.com/results?search_query=" + q;
                if (minuteMark > 0) {
                    url += "&t=" + std::to_string(minuteMark * 60) + "s";
                }
                openUrlInBrowser(g_preferredBrowserExe, url);
                replyUtf8 = "Открыла YouTube по запросу.";
            } else if ((wantsOpen || mentionsMusic) && mentionsPlaylist) {
                openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/feed/playlists");
                replyUtf8 = "Открыла плейлисты.";
            } else if ((wantsOpen || mentionsMusic) && (cmdSrc.find("ютуб музыка") != std::string::npos)) {
                openUrlInBrowser(g_preferredBrowserExe, "https://music.youtube.com/");
                replyUtf8 = "Открыла YouTube Music.";
            } else if (wantsOpen && mentionsBrowser) {
                openUrlInBrowser(g_preferredBrowserExe, "https://www.google.com/");
                replyUtf8 = "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                           "\xd0\xb1\xd1\x80\xd0\xb0\xd1\x83\xd0\xb7\xd0\xb5\xd1\x80 "
                           "\xd0\xb8\xd0\xb7 \xd0\xbd\xd0\xb0\xd1\x81\xd1\x82\xd1\x80\xd0\xbe\xd0\xb5\xd0\xba.";
            } else if (wantsClose && mentionsNotepad) {
                closeApp("notepad.exe");
                replyUtf8 =
                    "\xd0\x97\xd0\xb0\xd0\xba\xd1\x80\xd1\x8b\xd0\xbb\xd0\xb0 "
                    "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82.";
            } else if (!text.empty()) {
                replyUtf8 = ai.generateResponse(text, effectiveTone);
                if (replyUtf8 == "COMMAND_OPEN_NOTEPAD") {
                    openApp("notepad.exe");
                    replyUtf8 =
                        "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                        "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82. "
                        "\xd0\x95\xd1\x81\xd0\xbb\xd0\xb8 \xd1\x87\xd1\x82\xd0\xbe-"
                        "\xd1\x82\xd0\xbe \xe2\x80\x94 \xd0\xbc\xd0\xbe\xd0\xb3\xd1\x83 "
                        "\xd0\xbf\xd0\xbe\xd0\xbc\xd0\xbe\xd1\x87\xd1\x8c "
                        "\xd1\x81 \xd1\x82\xd0\xb5\xd0\xba\xd1\x81\xd1\x82\xd0\xbe\xd0\xbc.";
                } else if (replyUtf8 == "COMMAND_KILL_NOTEPAD") {
                    closeApp("notepad.exe");
                    replyUtf8 =
                        "\xd0\x97\xd0\xb0\xd0\xba\xd1\x80\xd1\x8b\xd0\xbb\xd0\xb0 "
                        "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82.";
                }
            } else {
                replyUtf8 = "\xd0\x9d\xd0\xb5\xd1\x82 \xd1\x82\xd0\xb5\xd0\xba\xd1\x81\xd1\x82\xd0\xb0 "
                           "\xd1\x81\xd0\xbe\xd0\xbe\xd0\xb1\xd1\x89\xd0\xb5\xd0\xbd\xd0\xb8\xd1\x8F.";
            }

            postW(MakeJsonTypingW(false));
            postW(MakeJsonReplyW(replyUtf8));
            return S_OK;
        }
    }

    const bool wantsOpen =
        (userText.find("otkroi") != std::string::npos) ||
        (userText.find("open") != std::string::npos) ||
        (userText.find("start") != std::string::npos) ||
        (userText.find("launch") != std::string::npos) ||
        (userText.find("включи") != std::string::npos) ||
        (userText.find("Включи") != std::string::npos) ||
        (userText.find("запусти") != std::string::npos) ||
        (userText.find("Запусти") != std::string::npos) ||
        (userText.find("открой") != std::string::npos) ||
        (userText.find("Открой") != std::string::npos);
    const bool wantsClose =
        (userText.find("zakroi") != std::string::npos) ||
        (userText.find("kill") != std::string::npos) ||
        (userText.find("закрой") != std::string::npos) ||
        (userText.find("Закрой") != std::string::npos);
    const bool mentionsNotepad =
        (userText.find("notepad") != std::string::npos) ||
        (userText.find("Notepad") != std::string::npos) ||
        (userText.find("блокнот") != std::string::npos) ||
        (userText.find("Блокнот") != std::string::npos);
    const bool mentionsBrowser =
        (userText.find("browser") != std::string::npos) ||
        (userText.find("Browser") != std::string::npos) ||
        (userText.find("браузер") != std::string::npos) ||
        (userText.find("Браузер") != std::string::npos);
    const bool mentionsAnime =
        (userText.find("anime") != std::string::npos) ||
        (userText.find("Anime") != std::string::npos) ||
        (userText.find("аниме") != std::string::npos) ||
        (userText.find("Аниме") != std::string::npos);
    const std::string userLower = ToLowerAscii(userText);
    const bool mentionsYoutube = (userLower.find("youtube") != std::string::npos) ||
                                (userText.find("YouTube") != std::string::npos) ||
                                (userText.find("ютуб") != std::string::npos) ||
                                (userText.find("Ютуб") != std::string::npos);
    const bool mentionsSpotify = (userLower.find("spotify") != std::string::npos) ||
                                (userText.find("спотиф") != std::string::npos);
    const bool mentionsPlaylist =
        (userLower.find("playlist") != std::string::npos) ||
        (userText.find("плейлист") != std::string::npos);
    const bool mentionsMusic = (userLower.find("music") != std::string::npos) ||
                              (userText.find("музык") != std::string::npos);
    const bool wantsLouder = (userText.find("громче") != std::string::npos);
    const bool wantsQuieter = (userText.find("тише") != std::string::npos);
    const bool wantsMute = (userText.find("выключи звук") != std::string::npos) ||
                           (userLower.find("mute") != std::string::npos);
    const bool wantsScrollDown = (userText.find("листай вниз") != std::string::npos) ||
                                 (userText.find("пролистай вниз") != std::string::npos) ||
                                 (userLower.find("scroll down") != std::string::npos);
    const bool wantsScrollUp = (userText.find("листай вверх") != std::string::npos) ||
                               (userText.find("пролистай вверх") != std::string::npos) ||
                               (userLower.find("scroll up") != std::string::npos);
    const bool wantsLike = (userText.find("поставь лайк") != std::string::npos) ||
                           (userLower.find("like video") != std::string::npos);
    const int minuteMark = ExtractMinuteNumber(userLower);
    const std::string animeHint = ExtractAnimeTitleHint(userText);
    const bool wantsSearch = (userLower.find("search") != std::string::npos) ||
                             (userText.find("поиск") != std::string::npos) ||
                             (userText.find("найди") != std::string::npos);
    const bool wantsFirstVideo = (userText.find("1 видео") != std::string::npos) ||
                                 (userText.find("первое видео") != std::string::npos) ||
                                 (userLower.find("first video") != std::string::npos);

    postW(MakeJsonTypingW(true));

    std::string replyUtf8;
    if (wantsOpen && mentionsNotepad) {
        openApp("notepad.exe");
        replyUtf8 = "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                     "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82. "
                     "\xd0\x95\xd1\x81\xd0\xbb\xd0\xb8 \xd0\xbd\xd0\xb0\xd0\xb4\xd0\xbe "
                     "\xe2\x80\x94 \xd0\xbf\xd0\xbe\xd0\xbc\xd0\xbe\xd0\xb3\xd1\x83 "
                     "\xd1\x81 \xd1\x82\xd0\xb5\xd0\xba\xd1\x81\xd1\x82\xd0\xbe\xd0\xbc.";
    } else if (wantsOpen && mentionsAnime) {
        if (!animeHint.empty()) {
            openUrlInBrowser(g_preferredBrowserExe,
                             std::string("https://www.google.com/search?btnI=1&q=site%3Ajut-su.net+") +
                                 UrlEncode(animeHint));
            replyUtf8 = "Открыла аниме по названию.";
        } else {
            openUrlInBrowser(g_preferredBrowserExe, "https://jut-su.net/");
            replyUtf8 = "Открыла сайт для просмотра аниме.";
        }
    } else if (wantsLouder) {
        volumeUp(6);
        replyUtf8 = "Сделала громче.";
    } else if (wantsQuieter) {
        volumeDown(6);
        replyUtf8 = "Сделала тише.";
    } else if (wantsMute) {
        volumeMuteToggle();
        replyUtf8 = "Переключила mute.";
    } else if (wantsScrollDown) {
        scrollPageDown(2);
        replyUtf8 = "Листнула вниз.";
    } else if (wantsScrollUp) {
        scrollPageUp(2);
        replyUtf8 = "Листнула вверх.";
    } else if (wantsLike) {
        focusAndPressEnter(10);
        replyUtf8 = "Пробую поставить лайк (best effort).";
    } else if ((wantsOpen || mentionsMusic) && mentionsSpotify) {
        openUrlInBrowser(g_preferredBrowserExe, "https://open.spotify.com/");
        replyUtf8 = "Открыла Spotify.";
    } else if ((wantsOpen || mentionsMusic) && mentionsYoutube && mentionsPlaylist) {
        openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/feed/playlists");
        replyUtf8 = "Открыла YouTube плейлисты.";
    } else if (wantsOpen && mentionsYoutube && !mentionsMusic && !mentionsPlaylist && !wantsSearch &&
               minuteMark <= 0) {
        openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
        replyUtf8 = "Открыла YouTube.";
    } else if ((wantsOpen || wantsSearch) && wantsFirstVideo) {
        std::string q = userText;
        ReplaceAllInPlace(q, "открой", " ");
        ReplaceAllInPlace(q, "запусти", " ");
        ReplaceAllInPlace(q, "включи", " ");
        ReplaceAllInPlace(q, "ютуб", " ");
        ReplaceAllInPlace(q, "youtube", " ");
        ReplaceAllInPlace(q, "1 видео", " ");
        ReplaceAllInPlace(q, "первое видео", " ");
        q = TrimAscii(q);
        if (q.empty()) {
            openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
            replyUtf8 = "Открыла YouTube. Напиши: найди <запрос> и открой 1 видео.";
        } else {
            openUrlInBrowser(g_preferredBrowserExe,
                             std::string("https://www.google.com/search?btnI=1&q=site%3Ayoutube.com%2Fwatch+") +
                                 UrlEncode(q));
            replyUtf8 = "Открыла первое видео по запросу.";
        }
    } else if (wantsOpen && mentionsYoutube && !wantsSearch && minuteMark <= 0) {
        openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/");
        replyUtf8 = "Открыла YouTube.";
    } else if ((wantsOpen || mentionsMusic || wantsSearch) && mentionsYoutube) {
        std::string q = UrlEncode(userText);
        std::string url = "https://www.youtube.com/results?search_query=" + q;
        if (minuteMark > 0) {
            url += "&t=" + std::to_string(minuteMark * 60) + "s";
        }
        openUrlInBrowser(g_preferredBrowserExe, url);
        replyUtf8 = "Открыла YouTube по запросу.";
    } else if ((wantsOpen || mentionsMusic) && mentionsPlaylist) {
        openUrlInBrowser(g_preferredBrowserExe, "https://www.youtube.com/feed/playlists");
        replyUtf8 = "Открыла плейлисты.";
    } else if ((wantsOpen || mentionsMusic) && (userText.find("ютуб музыка") != std::string::npos)) {
        openUrlInBrowser(g_preferredBrowserExe, "https://music.youtube.com/");
        replyUtf8 = "Открыла YouTube Music.";
    } else if (wantsOpen && mentionsBrowser) {
        openUrlInBrowser(g_preferredBrowserExe, "https://www.google.com/");
        replyUtf8 = "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                     "\xd0\xb1\xd1\x80\xd0\xb0\xd1\x83\xd0\xb7\xd0\xb5\xd1\x80 "
                     "\xd0\xb8\xd0\xb7 \xd0\xbd\xd0\xb0\xd1\x81\xd1\x82\xd1\x80\xd0\xbe\xd0\xb5\xd0\xba.";
    } else if (wantsClose && mentionsNotepad) {
        closeApp("notepad.exe");
        replyUtf8 = "\xd0\x97\xd0\xb0\xd0\xba\xd1\x80\xd1\x8b\xd0\xbb\xd0\xb0 "
                     "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82.";
    } else {
        replyUtf8 = ai.generateResponse(userText);
        if (replyUtf8 == "COMMAND_OPEN_NOTEPAD") {
            openApp("notepad.exe");
            replyUtf8 =
                "\xD0\x9E\xD1\x82\xD0\xBA\xD1\x80\xD1\x8B\xD0\xBB\xD0\xB0 "
                "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82. "
                "\xd0\x95\xd1\x81\xd0\xbb\xd0\xb8 \xd1\x87\xd1\x82\xd0\xbe-"
                "\xd1\x82\xd0\xbe \xe2\x80\x94 \xd0\xbc\xd0\xbe\xd0\xb3\xd1\x83 "
                "\xd0\xbf\xd0\xbe\xd0\xbc\xd0\xbe\xd1\x87\xd1\x8c "
                "\xd1\x81 \xd1\x82\xd0\xb5\xd0\xba\xd1\x81\xd1\x82\xd0\xbe\xd0\xbc.";
        } else if (replyUtf8 == "COMMAND_KILL_NOTEPAD") {
            closeApp("notepad.exe");
            replyUtf8 =
                "\xd0\x97\xd0\xb0\xd0\xba\xd1\x80\xd1\x8b\xd0\xbb\xd0\xb0 "
                "\xd0\xb1\xd0\xbb\xd0\xbe\xd0\xba\xd0\xbd\xd0\xbe\xd1\x82.";
        }
    }

    postW(MakeJsonTypingW(false));
    postW(MakeJsonReplyW(replyUtf8));
    return S_OK;
    } catch (...) {
        sender->PostWebMessageAsString(Utf8ToWideSafe(R"({"type":"reply","text":"Внутренняя ошибка. Попробуй ещё раз."})").c_str());
        return S_OK;
    }
}

