# yongle · 永乐

**yongle** is a multi-provider AI API client for Go, modeled after Simon's Rust crate [`baochuan`](https://github.com/thesimonharms/baochuan).

The name keeps the Zheng He theme: Zheng He's treasure voyages sailed under the **Yongle Emperor**. If ZhengHe brought AI access to Java and baochuan became the treasure fleet for Rust, **yongle** is the imperial commission that sends the fleet onward into Go.

## Features

- Unified `Provider` interface for chat, streaming, model listing, and TTS.
- Separate `AgentProvider` interface for stateful agent runtimes.
- Direct HTTP implementation using Go's standard `net/http` stack.
- OpenAI-compatible provider core for Nous Portal, OpenAI, DeepSeek, xAI/Grok, Mistral, OpenRouter, Moonshot/Kimi, Perplexity, Qwen compatible mode, GitHub Copilot, LM Studio, Ollama, and llama.cpp.
- Native adapters for Anthropic Claude, Google Gemini, and Cloudflare Workers AI.
- Hermes Agent `/v1/responses` support for non-streaming and streaming agent runs.
- Native builders for chat, text-to-speech, and agent requests.
- Server-Sent Events streaming parser exposed through Go iterator syntax.
- Extensible: implement `Provider`, `AgentProvider`, or wrap `OpenAICompatibleProvider` for more backends.

## Install

```bash
go get github.com/thesimonharms/yongle
```

For local development in this directory:

```bash
go test ./...
```

## Quickstart

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/thesimonharms/yongle"
)

func main() {
    provider := yongle.NewDeepSeekProvider(os.Getenv("DEEPSEEK_API_KEY"))

    req, err := yongle.NewChatRequest("deepseek-chat").
        User("Tell me about the treasure ships of Zheng He.").
        MaxTokens(512).
        Build()
    if err != nil {
        log.Fatal(err)
    }

    resp, err := provider.Chat(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Content())
}
```

## Streaming

```go
stream, err := provider.StreamChat(ctx, req)
if err != nil {
    log.Fatal(err)
}

for chunk, err := range stream {
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(chunk.DeltaContent())
}
```

## Provider constructors

| Constructor | Default base URL / notes |
|---|---|
| `NewOpenAIProvider` | `https://api.openai.com/v1` (also implements TTS) |
| `NewDeepSeekProvider` | `https://api.deepseek.com` |
| `NewXAIProvider` | `https://api.x.ai/v1` |
| `NewMistralProvider` | `https://api.mistral.ai/v1` |
| `NewOpenRouterProvider` | `https://openrouter.ai/api/v1` — set `HTTP-Referer` / `X-OpenRouter-Title` (or `X-Title`) via `WithHeader` for app attribution |
| `NewMoonshotProvider` | `https://api.moonshot.ai/v1` (global Kimi platform) |
| `NewMoonshotCNProvider` | `https://api.moonshot.cn/v1` (China platform) |
| `NewPerplexityProvider` | `https://api.perplexity.ai` |
| `NewNousProvider` | `https://inference-api.nousresearch.com/v1` |
| `NewQwenProvider` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `NewQwenCNProvider` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `NewCopilotProvider` | `https://api.githubcopilot.com` — sends `Copilot-Integration-Id: copilot-developer-cli` by default |
| `NewLMStudioProvider` | `http://localhost:1234/v1` |
| `NewOllamaProvider` | `http://localhost:11434/v1` |
| `NewLlamaCPPProvider` | caller-supplied base URL |
| `NewAnthropicProvider` | `https://api.anthropic.com/v1` (`x-api-key` + `anthropic-version: 2023-06-01`) |
| `NewGeminiProvider` | `https://generativelanguage.googleapis.com/v1beta` (`x-goog-api-key`) |
| `NewCloudflareProvider` | `https://api.cloudflare.com/client/v4` |
| `NewOpenAICompatibleProvider` | custom name + base URL |

Useful options:

```go
yongle.WithBaseURL("http://localhost:8080/v1")
yongle.WithHTTPClient(customClient)
yongle.WithHeader("HTTP-Referer", "https://example.com")
yongle.WithHeader("X-OpenRouter-Title", "My App")
```

For OpenAI reasoning / o-series models, prefer `MaxCompletionTokens` over `MaxTokens`:

```go
req, err := yongle.NewChatRequest("o4-mini").
    User("Explain Zheng He's voyages briefly.").
    MaxCompletionTokens(512).
    Build()
```

## Hermes Agent runs

Hermes Agent is exposed as a separate `AgentProvider` because it targets the higher-level OpenAI Responses-style `/v1/responses` endpoint rather than plain chat completions:

```go
agent := yongle.NewHermesAgentProvider(os.Getenv("API_SERVER_KEY"))

run, err := yongle.NewAgentRunRequest("hermes-agent", "What files are in this project?").
    Instructions("You are a helpful coding assistant.").
    Store(true).
    Build()
if err != nil { log.Fatal(err) }

resp, err := agent.Run(ctx, run)
if err != nil { log.Fatal(err) }
fmt.Println(resp.OutputText())
```

Streaming agent lifecycle events also use Go iterator syntax:

```go
stream, err := agent.StreamRun(ctx, run)
if err != nil { log.Fatal(err) }

for event, err := range stream {
    if err != nil { log.Fatal(err) }
    fmt.Print(event.OutputTextDelta())
}
```

## TTS

OpenAI-compatible chat providers return `ErrUnsupported` for TTS by default. `OpenAIProvider` implements `/audio/speech`:

```go
tts, err := yongle.NewTTSRequest("tts-1", "Hello from yongle.").
    Voice("nova").
    Format("mp3").
    Build()
if err != nil { log.Fatal(err) }

audio, err := yongle.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY")).TTS(ctx, tts)
if err != nil { log.Fatal(err) }
_ = os.WriteFile("hello.mp3", audio, 0644)
```

## Status

This Go port covers baochuan's core ergonomics with current upstream defaults: OpenAI-compatible providers (including Nous Portal), regional Moonshot/Qwen endpoints, Copilot integration headers, native Anthropic/Gemini/Cloudflare adapters, OpenAI `max_completion_tokens`, and Hermes `AgentProvider` support. Additional provider-specific features can be layered on the same `Provider` and `AgentProvider` interfaces without changing callers.

## License

MIT
