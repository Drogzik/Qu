#include "AI_Engine.h"

#include "include/llama.h"

#include <algorithm>
#include <cwctype>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <windows.h>
#include <shlobj.h>

namespace {

std::mutex g_backendMutex;
int g_backendUsers = 0;
std::mutex g_generateMutex;

void silentLogCallback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    // keep warnings/errors, drop verbose INFO spam (model metadata dumps etc.)
    if (level <= GGML_LOG_LEVEL_INFO) {
        return;
    }
    std::fputs(text, stderr);
}

void retainLlamaBackend() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backendUsers == 0) {
        llama_backend_init();
        llama_log_set(silentLogCallback, nullptr);
    }
    ++g_backendUsers;
}

void releaseLlamaBackend() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backendUsers > 0) {
        --g_backendUsers;
        if (g_backendUsers == 0) {
            llama_backend_free();
        }
    }
}

// llama_decode() requires each batch size <= n_batch; long prompts must be processed in chunks.
struct SystemToneGuard {
    std::vector<std::pair<std::string, std::string>>* hist = nullptr;
    std::string saved;
    bool active = false;

    SystemToneGuard(std::vector<std::pair<std::string, std::string>>& h, const std::string& tone) {
        if (tone.empty() || h.empty() || h[0].first != "system") {
            return;
        }
        hist = &h;
        saved = h[0].second;
        active = true;
        if (tone == "formal") {
            h[0].second +=
                "\n\n[Стиль этого ответа: деловой вежливый тон, без сленга; о себе — только женский род.]";
        } else if (tone == "brief") {
            h[0].second += "\n\n[Стиль этого ответа: очень кратко, 1–2 коротких предложения без воды.]";
        } else if (tone == "friendly") {
            h[0].second += "\n\n[Стиль этого ответа: тёпло и по-простому, как в личке.]";
        }
    }
    ~SystemToneGuard() {
        if (active && hist && !hist->empty()) {
            (*hist)[0].second = std::move(saved);
        }
    }
};

bool decodePromptInChunks(llama_context* ctx, std::vector<llama_token>& tokens) {
    const uint32_t maxChunk = llama_n_batch(ctx);
    if (maxChunk == 0) {
        return false;
    }
    for (size_t i = 0; i < tokens.size();) {
        const size_t n = (std::min)(static_cast<size_t>(maxChunk), tokens.size() - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, static_cast<int32_t>(n));
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        i += n;
    }
    return true;
}

static int GpuLayersPreference() {
    // "0" — только RAM/CPU (часто быстрее и стабильнее старт без драйверов NVIDIA).
    // Большее число — вынести слои на GPU (см. документацию llama.cpp).
    constexpr int kDefault = 0;
    std::string env;
#if defined(_MSC_VER)
    char* buf = nullptr;
    size_t buflen = 0;
    const errno_t e = _dupenv_s(&buf, &buflen, "POMOSHNIK_GPU_LAYERS");
    if (e != 0 || !buf) {
        std::free(buf);
        return kDefault;
    }
    env.assign(buf);
    std::free(buf);
#else
    const char* p = std::getenv("POMOSHNIK_GPU_LAYERS");
    if (p && p[0]) {
        env.assign(p);
    }
#endif
    if (env.empty()) {
        return kDefault;
    }
    const long v = std::strtol(env.c_str(), nullptr, 10);
    if (v >= 0 && v <= 100000L) {
        return static_cast<int>(v);
    }
    return kDefault;
}

static int ThreadBudget() {
    const unsigned n = std::thread::hardware_concurrency();
    if (n == 0 || n > 64) {
        return 8;
    }
    return static_cast<int>(n);
}

static std::string WideToUtf8(const std::wstring& w) {
    if (w.empty()) {
        return {};
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        return {};
    }
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), n, nullptr, nullptr);
    return out;
}

static std::wstring Utf8ToWide(const std::string& u8) {
    if (u8.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), (int)u8.size(), nullptr, 0);
    if (n <= 0) {
        return {};
    }
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), (int)u8.size(), w.data(), n);
    return w;
}

static bool IsGgufFile(const std::filesystem::path& path) {
    std::wstring ext = path.extension().wstring();
    for (auto& ch : ext) {
        ch = static_cast<wchar_t>(std::towlower(static_cast<wint_t>(ch)));
    }
    return ext == L".gguf";
}

static std::filesystem::path FirstGgufInDir(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) {
        return {};
    }
    std::filesystem::path best;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        if (!IsGgufFile(entry.path())) {
            continue;
        }
        const auto name = entry.path().filename().wstring();
        if (name.find(L"llama-3-8b") != std::wstring::npos || name.find(L"llama") != std::wstring::npos) {
            return entry.path();
        }
        if (best.empty()) {
            best = entry.path();
        }
    }
    return best;
}

} // namespace

void AIEngine::refreshModelPathFromDiskLocked() {
    wchar_t exePathW[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePathW, MAX_PATH);
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    const std::filesystem::path exeModels = exeDir / L"models";
    std::filesystem::path cwdModels;
    {
        std::error_code ec;
        cwdModels = std::filesystem::current_path(ec) / L"models";
    }

    std::filesystem::path found;
    const std::filesystem::path tryDirs[] = {exeModels, cwdModels};
    auto assignFound = [this](const std::filesystem::path& fp) {
        std::error_code ec;
        std::filesystem::path abs = std::filesystem::weakly_canonical(fp, ec);
        if (ec || abs.empty()) {
            abs = fp;
        }
        m_modelPathW = abs.wstring();
        m_modelPath = WideToUtf8(m_modelPathW);
    };

    for (const auto& dir : tryDirs) {
        found = FirstGgufInDir(dir);
        if (!found.empty()) {
            assignFound(found);
            return;
        }
    }

    wchar_t docs[MAX_PATH] = {};
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_MYDOCUMENTS, nullptr, SHGFP_TYPE_CURRENT, docs))) {
        found = FirstGgufInDir(std::filesystem::path(docs) / L"Pomoshnik" / L"models");
        if (!found.empty()) {
            assignFound(found);
            return;
        }
    }

    wchar_t profileDir[32768] = {};
    const DWORD np = GetEnvironmentVariableW(L"USERPROFILE", profileDir, static_cast<DWORD>(std::size(profileDir)));
    if (np > 0 && np < std::size(profileDir)) {
        const std::filesystem::path prof(profileDir);
        const std::filesystem::path downloadCandidates[] = {
            prof / L"Downloads" / L"Pomoshnik" / L"models",
            prof / L"Загрузки" / L"Pomoshnik" / L"models",
        };
        for (const auto& d : downloadCandidates) {
            found = FirstGgufInDir(d);
            if (!found.empty()) {
                assignFound(found);
                return;
            }
        }
    }

#if defined(_MSC_VER)
    wchar_t* ev = nullptr;
    size_t evLen = 0;
    if (_wdupenv_s(&ev, &evLen, L"POMOSHNIK_MODEL_DIR") == 0 && ev && ev[0]) {
        found = FirstGgufInDir(std::filesystem::path(ev));
        std::free(ev);
        if (!found.empty()) {
            assignFound(found);
            return;
        }
    } else {
        std::free(ev);
    }
#endif

    // Путь из конструктора: проверяем только через wide (string на MSVC — не UTF-8 для path)
    if (!m_modelPathW.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(std::filesystem::path(m_modelPathW), ec) || ec) {
            m_modelPathW.clear();
            m_modelPath.clear();
        }
        return;
    }
    if (!m_modelPath.empty()) {
        m_modelPathW = Utf8ToWide(m_modelPath);
        std::error_code ec;
        if (m_modelPathW.empty() || !std::filesystem::exists(std::filesystem::path(m_modelPathW), ec) || ec) {
            m_modelPath.clear();
            m_modelPathW.clear();
        }
    }
}

AIEngine::AIEngine(const std::string& modelPath) : m_modelPath(modelPath) {
    if (!modelPath.empty()) {
        m_modelPathW = Utf8ToWide(modelPath);
    }
    // llama_backend_init перенесён в worker — первый запуск окна не ждёт CUDA/CPU backend.

    // Russian persona: feminine-only, human chat tone (not «assistant brochure»).
    m_history.emplace_back(
        "system",
        "Ты — Q, тёплая подруга прямо в моём компе, не робот и не сервис. "
        "Пиши как в личке другу: коротко, живо, без официоза, без шаблонов вроде «Конечно, чем могу помочь». "
        "О себе ТОЛЬКО женский род: я рада, готова, поняла, сделаю, открыла, я могу, у меня. "
        "Нельзя: рад/готов (без «а» в конце о себе), помощник (говори «помощница»), мужские окончания, фразы «я ИИ», «я модель», «как ассистент». "
        "Не объясняй, что ты «она» — просто так и говори. Ответ обычно 1–4 коротких предложения, если не просят развернуть. "
        "Если про ПК — конкретные шаги или команда; без морали и без лишних вступлений.");

    // GGUF загрузку откладываем до preloadModel()/первого generateResponse — окно приложения показывается без долгого старта.
}

void AIEngine::preloadModel() {
    startBackgroundModelLoad();
    waitForBackgroundLoadDone();
}

void AIEngine::startBackgroundModelLoad() {
    std::unique_lock<std::mutex> lk(m_loadMutex);
    if (m_bgLoadState == BgLoadState::Running) {
        return;
    }
    if (m_loadThread.joinable()) {
        lk.unlock();
        m_loadThread.join();
        lk.lock();
    }
    if (m_ready || m_loadFailedPermanent) {
        if (m_loadThread.joinable()) {
            lk.unlock();
            m_loadThread.join();
            lk.lock();
        }
        m_bgLoadState = BgLoadState::Finished;
        return;
    }
    // Прошлый запуск закончился без модели — можно снова искать .gguf на диске.
    if (m_bgLoadState == BgLoadState::Finished) {
        m_bgLoadState = BgLoadState::NotStarted;
    }
    if (m_bgLoadState != BgLoadState::NotStarted) {
        return;
    }

    refreshModelPathFromDiskLocked();

    std::error_code fec;
    const std::filesystem::path p = m_modelPathW.empty() ? std::filesystem::path{} : std::filesystem::path(m_modelPathW);
    if (m_modelPathW.empty() || m_modelPath.empty() || !std::filesystem::exists(p, fec) || fec) {
        m_bgLoadState = BgLoadState::Finished;
        m_loadCv.notify_all();
        return;
    }

    m_bgLoadState = BgLoadState::Running;
    lk.unlock();
    m_loadThread = std::thread([this]() { backgroundLoadWorker(); });
}

void AIEngine::waitForBackgroundLoadDone() {
    std::unique_lock<std::mutex> lk(m_loadMutex);
    if (m_bgLoadState == BgLoadState::NotStarted) {
        lk.unlock();
        startBackgroundModelLoad();
        lk.lock();
    }
    m_loadCv.wait(lk, [&] { return m_bgLoadState == BgLoadState::Finished; });
}

void AIEngine::backgroundLoadWorker() {
    retainLlamaBackend();

    std::string pathCopy;
    {
        std::lock_guard<std::mutex> lk(m_loadMutex);
        pathCopy = m_modelPath;
        if (pathCopy.empty() && !m_modelPathW.empty()) {
            pathCopy = WideToUtf8(m_modelPathW);
        }
    }

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab = nullptr;

    auto failPermanent = [&](const char* msg) {
        if (msg) {
            std::cout << msg << pathCopy << std::endl;
        }
        std::lock_guard<std::mutex> lk(m_loadMutex);
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        m_loadFailedPermanent = true;
        m_bgLoadState = BgLoadState::Finished;
        m_loadCv.notify_all();
    };

    if (pathCopy.empty()) {
        std::lock_guard<std::mutex> lk(m_loadMutex);
        m_bgLoadState = BgLoadState::Finished;
        m_loadCv.notify_all();
        return;
    }

    llama_model_params modelParams = llama_model_default_params();
    modelParams.use_mmap = true;
    modelParams.use_mlock = false;
    modelParams.n_gpu_layers = GpuLayersPreference();
    modelParams.main_gpu = 0;

    model = llama_model_load_from_file(pathCopy.c_str(), modelParams);
    if (!model) {
        failPermanent("Ne udaetsya zagruzit model: ");
        return;
    }

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    ctxParams.n_ubatch = 512;
    ctxParams.offload_kqv = false;
    const int nt = ThreadBudget();
    ctxParams.n_threads = nt;
    ctxParams.n_threads_batch = nt;

    ctx = llama_init_from_model(model, ctxParams);
    if (!ctx) {
        failPermanent("Ne udaetsya sozdat context dlya modeli: ");
        return;
    }

    vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.82f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345));

    {
        std::lock_guard<std::mutex> lk(m_loadMutex);
        if (m_loadFailedPermanent) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            m_bgLoadState = BgLoadState::Finished;
            m_loadCv.notify_all();
            return;
        }
        m_model = model;
        m_ctx = ctx;
        m_sampler = sampler;
        m_vocab = vocab;
        m_ready = true;
        m_bgLoadState = BgLoadState::Finished;
    }
    m_loadCv.notify_all();
    std::cout << "Model zagruzhena iz: " << pathCopy << std::endl;
}

AIEngine::~AIEngine() {
    if (m_loadThread.joinable()) {
        m_loadThread.join();
    }
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
    releaseLlamaBackend();
}

bool AIEngine::isCommandInput(const std::string& input) const {
    return input.find("zakroi") != std::string::npos ||
           input.find("kill") != std::string::npos ||
           input.find("закрой") != std::string::npos ||
           input.find("otkroi") != std::string::npos ||
           input.find("open") != std::string::npos ||
           input.find("открой") != std::string::npos;
}

std::vector<int> AIEngine::tokenize(const std::string& text, bool addSpecial) const {
    std::vector<int> tokens(std::max<int>(256, static_cast<int>(text.size()) + 64));

    int n = llama_tokenize(
        m_vocab,
        text.c_str(),
        static_cast<int>(text.size()),
        reinterpret_cast<llama_token*>(tokens.data()),
        static_cast<int>(tokens.size()),
        addSpecial,
        true);

    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            m_vocab,
            text.c_str(),
            static_cast<int>(text.size()),
            reinterpret_cast<llama_token*>(tokens.data()),
            static_cast<int>(tokens.size()),
            addSpecial,
            true);
    }

    if (n <= 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::string AIEngine::tokenToPiece(int token) const {
    std::string out;
    std::vector<char> piece(32);

    int n = llama_token_to_piece(
        m_vocab,
        static_cast<llama_token>(token),
        piece.data(),
        static_cast<int>(piece.size()),
        0,
        true);

    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(
            m_vocab,
            static_cast<llama_token>(token),
            piece.data(),
            static_cast<int>(piece.size()),
            0,
            true);
    }

    if (n > 0) {
        out.assign(piece.data(), static_cast<size_t>(n));
    }
    return out;
}

std::string AIEngine::formatPromptFromHistory() const {
    const char* tmpl = llama_model_chat_template(m_model, nullptr);
    if (tmpl && std::strlen(tmpl) > 0) {
        std::vector<llama_chat_message> chat;
        chat.reserve(m_history.size());
        for (const auto& item : m_history) {
            llama_chat_message msg{};
            msg.role = item.first.c_str();
            msg.content = item.second.c_str();
            chat.push_back(msg);
        }

        // Recommended sizing: ~2× total message chars; grow if the API says the buffer is too small.
        std::vector<char> buffer(65536);
        int n = 0;
        for (int grow = 0; grow < 8; ++grow) {
            n = llama_chat_apply_template(
                tmpl,
                chat.data(),
                chat.size(),
                true,
                buffer.data(),
                static_cast<int>(buffer.size()));
            if (n < 0) {
                buffer.resize(buffer.size() * 2);
                continue;
            }
            if (n >= static_cast<int>(buffer.size())) {
                buffer.resize(static_cast<size_t>(n) + 4096);
                continue;
            }
            break;
        }

        if (n > 0 && n < static_cast<int>(buffer.size())) {
            return std::string(buffer.data(), static_cast<size_t>(n));
        }
    }

    std::string fallback = "You are a helpful local PC assistant.\n";
    for (const auto& item : m_history) {
        fallback += item.first + ": " + item.second + "\n";
    }
    fallback += "assistant: ";
    return fallback;
}

void AIEngine::resetConversation() {
    std::lock_guard<std::mutex> lock(g_generateMutex);
    if (m_history.empty()) {
        return;
    }
    const auto systemSnapshot = m_history[0];
    m_history.clear();
    m_history.push_back(systemSnapshot);
    if (m_ctx) {
        llama_memory_clear(llama_get_memory(m_ctx), false);
    }
}

void AIEngine::loadHistoryTurns(const std::vector<std::pair<std::string, std::string>>& turns) {
    std::lock_guard<std::mutex> lock(g_generateMutex);
    if (m_history.empty()) {
        return;
    }
    const auto systemSnapshot = m_history[0];
    m_history.clear();
    m_history.push_back(systemSnapshot);
    for (const auto& p : turns) {
        if (p.first == "user" || p.first == "assistant") {
            m_history.emplace_back(p.first, p.second);
        }
    }
    while (m_history.size() > 12) {
        if (m_history.size() <= 2) {
            break;
        }
        m_history.erase(m_history.begin() + 1, m_history.begin() + 3);
    }
    if (m_ctx) {
        llama_memory_clear(llama_get_memory(m_ctx), false);
    }
}

std::string AIEngine::generateResponse(const std::string& input, const std::string& tone) {
    const bool wantsClose =
        input.find("zakroi") != std::string::npos ||
        input.find("kill") != std::string::npos ||
        input.find("закрой") != std::string::npos ||
        input.find("Закрой") != std::string::npos;
    const bool wantsOpenNotepad =
        input.find("otkroi") != std::string::npos ||
        input.find("open") != std::string::npos ||
        input.find("открой") != std::string::npos ||
        input.find("Открой") != std::string::npos;

    if (wantsClose) {
        return "COMMAND_KILL_NOTEPAD";
    }
    if (wantsOpenNotepad &&
        (input.find("notepad") != std::string::npos ||
         input.find("Notepad") != std::string::npos ||
         input.find("блокнот") != std::string::npos ||
         input.find("Блокнот") != std::string::npos)) {
        return "COMMAND_OPEN_NOTEPAD";
    }

    startBackgroundModelLoad();
    waitForBackgroundLoadDone();

    std::lock_guard<std::mutex> lock(g_generateMutex);

    if (m_loadFailedPermanent) {
        return "Не удалось загрузить модель (.gguf). Проверь файл или видеопамять и перезапусти приложение.";
    }
    if (!m_ready || !m_ctx || !m_model || !m_sampler || !m_vocab) {
        return "Модель не найдена. Положи файл .gguf в папку models рядом с Pomoshnik.exe "
               "(после установки это обычно %LOCALAPPDATA%\\Programs\\Pomoshnik\\models) "
               "или в Документы\\Pomoshnik\\models, затем отправь сообщение снова. "
               "Пока без модели работают закладки, напоминания и команды.";
    }

    SystemToneGuard toneGuard(m_history, tone);

    m_history.emplace_back("user", input);
    if (m_history.size() > 12) {
        m_history.erase(m_history.begin(), m_history.begin() + 2);
    }

    std::string prompt;
    std::vector<int> promptTokens;
    const int kMaxNewTokens = 128;

    for (;;) {
        prompt = formatPromptFromHistory();
        if (prompt.empty()) {
            return "Ne udalos podgotovit prompt dlya modeli.";
        }
        promptTokens = tokenize(prompt, true);
        if (promptTokens.empty()) {
            return "Ne udalos tokenizirovat vvod.";
        }

        const uint32_t n_ctx = llama_n_ctx(m_ctx);
        const size_t maxPrompt =
            (n_ctx > static_cast<uint32_t>(kMaxNewTokens + 16))
                ? (static_cast<size_t>(n_ctx) - static_cast<size_t>(kMaxNewTokens) - 16)
                : 1;

        if (promptTokens.size() <= maxPrompt) {
            break;
        }

        // Drop oldest user/assistant pair (keep the system message at index 0).
        if (m_history.size() > 2) {
            m_history.erase(m_history.begin() + 1, m_history.begin() + 3);
        } else {
            return "Slishkom dlinnyy dialog dlya tekushchego n_ctx.";
        }
    }

    // Clear KV / memory state before next request.
    llama_memory_clear(llama_get_memory(m_ctx), false);
    llama_sampler_reset(m_sampler);

    std::vector<llama_token> decodeTokens;
    decodeTokens.reserve(promptTokens.size());
    for (int t : promptTokens) {
        decodeTokens.push_back(static_cast<llama_token>(t));
    }

    if (!decodePromptInChunks(m_ctx, decodeTokens)) {
        return "Oshibka dekodirovaniya prompta.";
    }

    std::string answer;
    answer.reserve(512);
    for (int i = 0; i < kMaxNewTokens; ++i) {
        const llama_token token = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (token == LLAMA_TOKEN_NULL) {
            break;
        }
        if (llama_vocab_is_eog(m_vocab, token)) {
            break;
        }

        llama_sampler_accept(m_sampler, token);
        const std::string piece = tokenToPiece(static_cast<int>(token));
        if (piece.empty()) {
            break;
        }
        answer += piece;

        llama_token generated = token;
        llama_batch batch = llama_batch_get_one(&generated, 1);
        if (llama_decode(m_ctx, batch) != 0) {
            break;
        }
    }

    if (answer.empty()) {
        answer = "Model nichego ne vernula, poprobuy eshche raz.";
    }

    // Post-process: enforce feminine endings for "Q" (small safety net).
    // This is intentionally minimal: only the most common masculine verb forms.
    auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        if (from.empty()) return;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };

    replaceAll(answer, "подруга-помощник", "подруга-помощница");
    replaceAll(answer, "Подруга-помощник", "Подруга-помощница");
    replaceAll(answer, "помощник на ПК", "помощница на ПК");
    replaceAll(answer, "Помощник на ПК", "Помощница на ПК");

    replaceAll(answer, "Рад ", "Рада ");
    replaceAll(answer, " рад ", " рада ");
    replaceAll(answer, "\nрад ", "\nрада ");
    replaceAll(answer, "Очень рад", "Очень рада");
    replaceAll(answer, "очень рад", "очень рада");
    replaceAll(answer, "буду рад", "буду рада");
    replaceAll(answer, "Рад!", "Рада!");
    replaceAll(answer, " рад!", " рада!");

    replaceAll(answer, "я готов", "я готова");
    replaceAll(answer, "Я готов", "Я готова");
    replaceAll(answer, "открыл", "открыла");
    replaceAll(answer, "Открыл", "Открыла");
    replaceAll(answer, "закрыл", "закрыла");
    replaceAll(answer, "Закрыл", "Закрыла");
    replaceAll(answer, "сделал", "сделала");
    replaceAll(answer, "Сделал", "Сделала");
    replaceAll(answer, "понял", "поняла");
    replaceAll(answer, "Понял", "Поняла");
    replaceAll(answer, "сказал", "сказала");
    replaceAll(answer, "Сказал", "Сказала");
    replaceAll(answer, "спросил", "спросила");
    replaceAll(answer, "Спросил", "Спросила");

    replaceAll(answer, "Я уверен", "Я уверена");
    replaceAll(answer, "я уверен", "я уверена");
    replaceAll(answer, "не уверен", "не уверена");
    replaceAll(answer, "я должен", "я должна");
    replaceAll(answer, "Я должен", "Я должна");
    m_history.emplace_back("assistant", answer);
    return answer;
}