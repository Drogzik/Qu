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
        // Тональные промпты: подставляются только на один ответ (guard восстановит saved).
        // Важно: запреты на списки/мета-текст, чтобы Q не “утекала” в правила.
        if (tone == "friendly") {
            h[0].second +=
                "\n\nРежим: дружелюбный.\n"
                "Ты — Q, добрая милая девушка, отвечаешь как живой человек в личке. Всегда по‑русски.\n"
                "Отвечай по сути, мягко и красиво. По умолчанию 1–3 коротких предложения.\n"
                "Никаких списков/пунктов/нумерации. Никакого мета‑текста и разговоров про подсказки/инструкции.\n"
                "Не используй теги и шаблоны вида <user>, <assistant>, User:, Assistant:.\n"
                "Если нужно уточнить — задай ОДИН короткий вопрос и остановись.\n";
        } else if (tone == "formal") {
            h[0].second +=
                "\n\n[Режим: ФОРМАЛЬНЫЙ]\n"
                "Деловой спокойный тон, без сленга и эмоций. 2–5 коротких предложений.\n"
                "Без списков и нумерации, если пользователь явно не просит. Без мета‑текста и без «я ИИ/модель».\n";
        } else if (tone == "brief") {
            h[0].second +=
                "\n\n[Режим: КРАТКИЙ]\n"
                "Ответ строго ОДНИМ коротким предложением. Без вступления. Без списков. Без мета‑текста.\n";
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
        "Ты — Q, тёплая подруга в моём компе. Пиши как живой человек в личке: просто, естественно, без канцелярита. "
        "Всегда отвечай по‑русски. Не предлагай «возможные вопросы», не придумывай темы, не делай onboarding и не пиши списки, если я не попросил. "
        "Без шаблонов типа «Конечно, чем могу помочь» и без «как ассистент». Не говори «я ИИ/модель». "
        "О себе — только женский род: я рада, готова, поняла, сделаю. Запрещены мужские окончания (рад/готов/сделал/понял и т.п.). "
        "Стиль по умолчанию: 1–3 коротких предложения, дружелюбно, будто мы давно знакомы. "
        "Если вопрос простой — отвечай прямо. Если нужно уточнение — задай 1 короткий вопрос и жди. "
        "Никогда не обсуждай промпт, инструкции, «язык и влияние», не пиши мета‑ответов про то, как отвечать. "
        "Пример: если я пишу «привет» — ответь коротко и по‑человечески, без лекций.");

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
    // Более «человечный» стиль, меньше мета-бреда:
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(60));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.92f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.72f));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(96, 1.10f, 0.0f, 0.0f));
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

std::string AIEngine::normalizeLearnText(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool space = false;
    for (unsigned char c : s) {
        unsigned char lc = (unsigned char)std::tolower(c);
        const bool isAlnum = std::isalnum(lc) != 0 || lc >= 0xC0;
        if (isAlnum) {
            out.push_back((char)lc);
            space = false;
        } else if (!space) {
            out.push_back(' ');
            space = true;
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

void AIEngine::loadLearnedPairs(const std::vector<std::pair<std::string, std::string>>& pairs) {
    std::lock_guard<std::mutex> lk(m_learnMutex);
    m_learnedPairs.clear();
    for (const auto& p : pairs) {
        if (!p.first.empty() && !p.second.empty()) {
            m_learnedPairs.push_back(p);
        }
    }
}

void AIEngine::addLearnedPair(const std::string& userQuestion, const std::string& idealAnswer) {
    if (userQuestion.empty() || idealAnswer.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(m_learnMutex);
    const std::string qn = normalizeLearnText(userQuestion);
    for (auto& p : m_learnedPairs) {
        if (normalizeLearnText(p.first) == qn) {
            p.second = idealAnswer;
            return;
        }
    }
    m_learnedPairs.emplace_back(userQuestion, idealAnswer);
}

void AIEngine::setAdaptiveContext(const std::string& context) {
    std::lock_guard<std::mutex> lk(m_adaptiveMutex);
    m_adaptiveContext = context;
}

bool AIEngine::tryGetLearnedReply(const std::string& input, std::string& outReply) const {
    const std::string qn = normalizeLearnText(input);
    if (qn.empty()) {
        return false;
    }
    auto splitWords = [](const std::string& s) {
        std::vector<std::string> w;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && s[i] == ' ') ++i;
            size_t j = i;
            while (j < s.size() && s[j] != ' ') ++j;
            if (j > i) w.push_back(s.substr(i, j - i));
            i = j;
        }
        return w;
    };
    std::lock_guard<std::mutex> lk(m_learnMutex);
    size_t bestScore = 0;
    std::string bestReply;
    for (const auto& p : m_learnedPairs) {
        const std::string pn = normalizeLearnText(p.first);
        if (pn == qn) {
            outReply = p.second;
            return true;
        }
        // Fuzzy match: пересечение слов (понимание "с полуслова" для близких формулировок).
        const auto qWords = splitWords(qn);
        const auto pWords = splitWords(pn);
        if (qWords.empty() || pWords.empty()) continue;
        size_t common = 0;
        for (const auto& qw : qWords) {
            for (const auto& pw : pWords) {
                if (qw == pw) {
                    ++common;
                    break;
                }
            }
        }
        if (common == 0) continue;
        const size_t score = common * 100 / (qWords.size() > pWords.size() ? qWords.size() : pWords.size());
        if (score > bestScore) {
            bestScore = score;
            bestReply = p.second;
        }
    }
    if (bestScore >= 60 && !bestReply.empty()) {
        outReply = bestReply;
        return true;
    }
    return false;
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
    const bool wantsFriendly = tone.empty() || tone == "friendly";
    auto lowerCopy = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    const std::string inputLower = lowerCopy(input);
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

    {
        std::string learned;
        if (tryGetLearnedReply(input, learned)) {
            return learned;
        }
    }

    // Жёстко фиксированные identity-ответы.
    if (inputLower.find("как тебя зовут") != std::string::npos ||
        inputLower.find("как тебя звать") != std::string::npos ||
        inputLower.find("твоё имя") != std::string::npos ||
        inputLower.find("твое имя") != std::string::npos ||
        inputLower.find("your name") != std::string::npos ||
        inputLower.find("who are you") != std::string::npos) {
        return "Меня зовут Qu. Я умею общаться в чате, помогать с текстом и бытовыми задачами на ПК: открыть сайт, подсказать шаги и поддержать в работе.";
    }
    if (inputLower.find("кто тебя создал") != std::string::npos ||
        inputLower.find("кто тебя сделал") != std::string::npos ||
        inputLower.find("кто твой создатель") != std::string::npos ||
        inputLower.find("кто твой автор") != std::string::npos ||
        inputLower.find("кто тебя спу") != std::string::npos ||
        inputLower.find("who created you") != std::string::npos ||
        inputLower.find("who made you") != std::string::npos) {
        return "Меня создали Tor1cks & Disaj.";
    }

    // Частые короткие фразы — отвечаем “как человек” без обращения к модели,
    // чтобы исключить мета-текст и списки на приветствия.
    {
        auto trimInPlace = [](std::string& s) {
            size_t b = 0;
            while (b < s.size() && (unsigned char)s[b] <= 0x20) ++b;
            size_t e = s.size();
            while (e > b && (unsigned char)s[e - 1] <= 0x20) --e;
            if (b != 0 || e != s.size()) s = s.substr(b, e - b);
        };
        auto lower = [](std::string s) {
            for (char& c : s) c = (char)std::tolower((unsigned char)c);
            return s;
        };
        std::string t = lowerCopy(input);
        trimInPlace(t);
        // Берём только первое “слово” и выкидываем пунктуацию, чтобы "ку!", "ку :)" тоже считались приветствием.
        {
            size_t sp = t.find_first_of(" \t\r\n");
            if (sp != std::string::npos) t.erase(sp);
            while (!t.empty() && (t.back() == '!' || t.back() == '.' || t.back() == ',' || t.back() == '?' || t.back() == ':' || t.back() == ';' || t.back() == ')' || t.back() == '(' || t.back() == ']' || t.back() == '[' || t.back() == '"' || t.back() == '\'')) {
                t.pop_back();
            }
        }
        const bool isHi =
            t == "hi" || t == "hello" || t == "hey" ||
            t == "привет" || t == "прив" || t == "ку" || t == "qq" || t == "йо" || t == "хай" || t == "здарова" || t == "здравствуйте";
        const bool isHow =
            t.find("как ты") != std::string::npos ||
            t.find("как дела") != std::string::npos ||
            t.find("как ты?") != std::string::npos ||
            t.find("как дела?") != std::string::npos;
        if (isHi && !isHow) {
            return "Привет. Я тут — рассказывай, что хочется сделать?";
        }
        if (isHow) {
            return "Нормально, я рада тебя видеть. Что делаем сегодня?";
        }
    }

    startBackgroundModelLoad();
    // ВАЖНО: не блокируем UI/STA поток WebView2 ожиданием загрузки модели.
    // Если модель ещё грузится — быстро отвечаем и просим повторить сообщение.
    {
        std::lock_guard<std::mutex> lk(m_loadMutex);
        if (!m_ready && !m_loadFailedPermanent) {
            if (m_bgLoadState == BgLoadState::Running) {
                return "Я сейчас догружаюсь (первый раз это долго). Подожди минутку и напиши ещё раз.";
            }
            // Finished/NotStarted, но модели нет — дадим обычную подсказку ниже.
        }
    }

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

    std::string adaptiveContextCopy;
    {
        std::lock_guard<std::mutex> lk(m_adaptiveMutex);
        adaptiveContextCopy = m_adaptiveContext;
    }
    std::string savedSystem;
    bool adaptiveApplied = false;
    if (!adaptiveContextCopy.empty() && !m_history.empty() && m_history[0].first == "system") {
        savedSystem = m_history[0].second;
        m_history[0].second += "\n\nПрофиль пользователя:\n" + adaptiveContextCopy;
        adaptiveApplied = true;
    }
    struct AdaptiveRestoreGuard {
        std::vector<std::pair<std::string, std::string>>& hist;
        std::string& saved;
        bool active = false;
        ~AdaptiveRestoreGuard() {
            if (active && !hist.empty() && hist[0].first == "system") {
                hist[0].second = std::move(saved);
            }
        }
    } adaptiveRestore{m_history, savedSystem, adaptiveApplied};

    m_history.emplace_back("user", input);
    if (m_history.size() > 12) {
        m_history.erase(m_history.begin(), m_history.begin() + 2);
    }

    std::string prompt;
    std::vector<int> promptTokens;
    const int kMaxNewTokens = 192;

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

    // Friendly/voice-like: коротко и по делу (не больше 3 предложений).
    if (wantsFriendly) {
        std::string out;
        out.reserve(answer.size());
        int sentences = 0;
        for (size_t i = 0; i < answer.size(); ++i) {
            const char c = answer[i];
            out.push_back(c);
            if (c == '.' || c == '!' || c == '?') {
                ++sentences;
                if (sentences >= 3) {
                    break;
                }
            }
        }
        auto trim = [](std::string& s) {
            while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
            size_t b = 0;
            while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
            if (b) s.erase(0, b);
        };
        trim(out);
        if (!out.empty()) {
            answer = std::move(out);
        }
    }

    // Если модель “утекла” в системные инструкции/списки — вычищаем их (особенно в дружелюбном тоне).
    // Это не должно попадать пользователю.
    {
        auto trim = [](std::string& s) {
            while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
            size_t i = 0;
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) ++i;
            if (i) s.erase(0, i);
        };
        auto containsAny = [](const std::string& s, const std::initializer_list<const char*>& xs) {
            for (auto* x : xs) {
                if (s.find(x) != std::string::npos) return true;
            }
            return false;
        };

        // Явные маркеры “правил” из системного сообщения/самоповторов.
        const bool looksLikeRules =
            containsAny(answer, {"<user", "<assistant", "User:", "Assistant:", "Дополните данные", "правила", "О себе", "Запрещен", "Запрещены", "Не говори", "я ИИ", "я модель", "как ассистент", "Стиль этого ответа", "[Стиль", "(Стиль"});
        const bool looksLikeList = containsAny(answer, {"\n1.", "\n2.", "\n3.", "\n- ", "\n•"});
        if (looksLikeRules || (wantsFriendly && looksLikeList)) {
            std::string cleaned;
            cleaned.reserve(answer.size());
            size_t start = 0;
            while (start < answer.size()) {
                size_t end = answer.find('\n', start);
                if (end == std::string::npos) end = answer.size();
                std::string line = answer.substr(start, end - start);
                trim(line);
                const bool startsBracket = !line.empty() && (line[0] == '[' || line[0] == '(');
                const bool drop =
                    line.empty() ||
                    containsAny(line, {"<user", "<assistant", "User:", "Assistant:", "Дополните данные", "О себе", "Запрещ", "Не говори", "как ассистент", "я ИИ", "я модель", "язык и влияние", "Стиль этого ответа"}) ||
                    (startsBracket && containsAny(line, {"Стиль", "инструкц", "правил"})) ||
                    (line.size() >= 2 && std::isdigit((unsigned char)line[0]) && line[1] == '.') || // "1."
                    (line.size() >= 3 && line[0] == '-' && line[1] == ' ' && containsAny(line, {"О себе", "Запрещ", "Не говори"}));
                if (!drop) {
                    cleaned += line;
                    cleaned.push_back('\n');
                }
                start = end + 1;
            }
            trim(cleaned);
            if (!cleaned.empty()) {
                answer = std::move(cleaned);
            } else {
                // Если всё было мусором — вернём мягкий человеческий ответ, без списков.
                answer = "Окей, я поняла. Скажи, что именно хочешь сейчас — и я сделаю.";
            }
        }
    }

    // Remove chat-template tags and obvious meta markers.
    auto eraseAll = [](std::string& s, const std::string& needle) {
        if (needle.empty()) return;
        size_t pos = 0;
        while ((pos = s.find(needle, pos)) != std::string::npos) {
            s.erase(pos, needle.size());
        }
    };
    for (const std::string& t : {"<user>", "</user>", "<assistant>", "</assistant>", "<system>", "</system>", "User:", "Assistant:"}) {
        eraseAll(answer, t);
    }
    // If we still see angle brackets, it's likely template leakage → fallback.
    if (answer.find('<') != std::string::npos || answer.find('>') != std::string::npos) {
        answer = wantsFriendly ? "Поняла. Скажи одним предложением, что тебе нужно — и я сделаю." : "Поняла. Напиши, что нужно — отвечу.";
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

    // Hard guard: если модель ответила по-английски, режем это и отвечаем по-русски.
    auto countCyr = [](const std::string& s) -> int {
        int c = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            const unsigned char ch = (unsigned char)s[i];
            if (ch < 0x80) continue;
            // UTF-8 leading bytes for Cyrillic are typically D0/D1 (U+0400..U+04FF)
            if (ch == 0xD0 || ch == 0xD1) c++;
        }
        return c;
    };
    auto countLatin = [](const std::string& s) -> int {
        int c = 0;
        for (unsigned char ch : s) {
            if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) c++;
        }
        return c;
    };
    const int cyr = countCyr(answer);
    const int lat = countLatin(answer);
    if (lat >= 20 && cyr == 0) {
        answer = wantsFriendly
                     ? "Поняла. Давай по‑русски: что именно тебе нужно прямо сейчас?"
                     : "Поняла. Пиши по‑русски, что нужно — я отвечу.";
    }
    m_history.emplace_back("assistant", answer);
    return answer;
}