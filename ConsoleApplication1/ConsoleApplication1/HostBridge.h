#pragma once

#include <windows.h>
#include <wrl.h>
#include <WebView2.h>

#include <string>

class AIEngine;

// Loads ui/app_shell.html next to the executable (visual layer).
std::wstring LoadShellHtmlW();

// Handles postMessage JSON and plain-text commands from the WebView (system / host protocol).
HRESULT HandleHostWebMessage(ICoreWebView2* sender, ICoreWebView2WebMessageReceivedEventArgs* args, AIEngine& ai,
                             const std::wstring& userDataDir);
