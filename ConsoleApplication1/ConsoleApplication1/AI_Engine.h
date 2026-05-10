#pragma once
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;

class AIEngine {
public:
    AIEngine(const std::string& modelPath);
    ~AIEngine();

    AIEngine(const AIEngine&) = delete;
    AIEngine& operator=(const AIEngine&) = delete;

    std::string generateResponse(const std::string& input, const std::string& tone = {});

    // Load GGUF weights (heavy). GUI defers until first inference; console mode should call this once up front.
    void preloadModel();

    // Старт фоновой загрузки модели (не блокирует UI). Безопасно вызывать несколько раз.
    void startBackgroundModelLoad();

    // Clears chat with the model but keeps the system persona.
    void resetConversation();

    // Restores user/assistant turns into server-side history (e.g. when switching a saved thread).
    // Pairs must alternate user/assistant; invalid entries are skipped.
    void loadHistoryTurns(const std::vector<std::pair<std::string, std::string>>& turns);
    void loadLearnedPairs(const std::vector<std::pair<std::string, std::string>>& pairs);
    void addLearnedPair(const std::string& userQuestion, const std::string& idealAnswer);
    void setAdaptiveContext(const std::string& context);

private:
    enum class BgLoadState { NotStarted, Running, Finished };

    void waitForBackgroundLoadDone();
    void backgroundLoadWorker();
    // Обновить m_modelPath с диска (exe\models, документы и т.д.). Вызывать под m_loadMutex.
    void refreshModelPathFromDiskLocked();

    std::string formatPromptFromHistory() const;
    std::vector<int> tokenize(const std::string& text, bool addSpecial) const;
    std::string tokenToPiece(int token) const;
    bool isCommandInput(const std::string& input) const;
    bool tryGetLearnedReply(const std::string& input, std::string& outReply) const;
    static std::string normalizeLearnText(const std::string& s);

    mutable std::mutex m_loadMutex;
    std::condition_variable m_loadCv;
    std::thread m_loadThread;
    BgLoadState m_bgLoadState = BgLoadState::NotStarted;

    bool m_loadFailedPermanent = false; // после ошибки llama в этой сессии не повторяем
    bool m_ready = false;
    std::wstring m_modelPathW; // для exists() на Windows (UTF-8 string ≠ путь в filesystem)
    std::string m_modelPath;   // UTF-8 для llama_model_load_from_file
    llama_model* m_model = nullptr;
    llama_context* m_ctx = nullptr;
    llama_sampler* m_sampler = nullptr;
    const llama_vocab* m_vocab = nullptr;
    std::vector<std::pair<std::string, std::string>> m_history;
    mutable std::mutex m_learnMutex;
    std::vector<std::pair<std::string, std::string>> m_learnedPairs;
    mutable std::mutex m_adaptiveMutex;
    std::string m_adaptiveContext;
};