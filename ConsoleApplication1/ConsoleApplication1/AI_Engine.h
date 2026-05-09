#pragma once
#include <string>
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

    // Clears chat with the model but keeps the system persona.
    void resetConversation();

    // Restores user/assistant turns into server-side history (e.g. when switching a saved thread).
    // Pairs must alternate user/assistant; invalid entries are skipped.
    void loadHistoryTurns(const std::vector<std::pair<std::string, std::string>>& turns);

private:
    std::string formatPromptFromHistory() const;
    std::vector<int> tokenize(const std::string& text, bool addSpecial) const;
    std::string tokenToPiece(int token) const;
    bool isCommandInput(const std::string& input) const;

    bool m_ready = false;
    std::string m_modelPath;
    llama_model* m_model = nullptr;
    llama_context* m_ctx = nullptr;
    llama_sampler* m_sampler = nullptr;
    const llama_vocab* m_vocab = nullptr;
    std::vector<std::pair<std::string, std::string>> m_history;
};