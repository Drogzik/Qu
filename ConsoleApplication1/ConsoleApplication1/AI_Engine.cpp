#include "AI_Engine.h"

#include "include/llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

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

} // namespace

AIEngine::AIEngine(const std::string& modelPath) : m_modelPath(modelPath) {
    retainLlamaBackend();

    // Russian persona: feminine-only, human chat tone (not «assistant brochure»).
    m_history.emplace_back(
        "system",
        "Ты — Q, тёплая подруга прямо в моём компе, не робот и не сервис. "
        "Пиши как в личке другу: коротко, живо, без официоза, без шаблонов вроде «Конечно, чем могу помочь». "
        "О себе ТОЛЬКО женский род: я рада, готова, поняла, сделаю, открыла, я могу, у меня. "
        "Нельзя: рад/готов (без «а» в конце о себе), помощник (говори «помощница»), мужские окончания, фразы «я ИИ», «я модель», «как ассистент». "
        "Не объясняй, что ты «она» — просто так и говори. Ответ обычно 1–4 коротких предложения, если не просят развернуть. "
        "Если про ПК — конкретные шаги или команда; без морали и без лишних вступлений.");

    llama_model_params modelParams = llama_model_default_params();
    // Reduce RAM: map model from disk; keep it unlocked by default.
    modelParams.use_mmap = true;
    modelParams.use_mlock = false;

    // 1660 SUPER has ~6 GB VRAM; offload a moderate amount by default.
    // More layers => more VRAM, usually less RAM pressure.
    modelParams.n_gpu_layers = 28;
    modelParams.main_gpu = 0;

    m_model = llama_model_load_from_file(m_modelPath.c_str(), modelParams);
    if (!m_model) {
        std::cout << "Ne udaetsya zagruzit model: " << modelPath << std::endl;
        return;
    }

    llama_context_params ctxParams = llama_context_default_params();
    // Enough room for Llama-3 chat template + Russian system message + a few turns.
    // (Previously n_batch=128 forced us to truncate the prompt mid-sequence → garbage output.)
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    ctxParams.n_ubatch = 512;
    // KV на CPU для стабильности при повторных генерациях.
    ctxParams.offload_kqv = false;
    ctxParams.n_threads = 8;
    ctxParams.n_threads_batch = 8;

    m_ctx = llama_init_from_model(m_model, ctxParams);
    if (!m_ctx) {
        std::cout << "Ne udaetsya sozdat context dlya modeli." << std::endl;
        llama_model_free(m_model);
        m_model = nullptr;
        return;
    }

    m_vocab = llama_model_get_vocab(m_model);

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    m_sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(0.90f, 1));
    // Slightly warmer sampling for more natural spoken Russian.
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.82f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(12345));

    m_ready = true;
    std::cout << "Model zagruzhena iz: " << modelPath << std::endl;
}

AIEngine::~AIEngine() {
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
    std::lock_guard<std::mutex> lock(g_generateMutex);
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

    if (!m_ready || !m_ctx || !m_model || !m_sampler || !m_vocab) {
        return "Model ne gotova. Prover put k GGUF i perezapusti programmu.";
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