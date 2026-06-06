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

- `NewOpenAIProvider(apiKey, opts...)`
- `NewNousProvider(apiKey, opts...)`
- `NewDeepSeekProvider(apiKey, opts...)`
- `NewXAIProvider(apiKey, opts...)`
- `NewMistralProvider(apiKey, opts...)`
- `NewOpenRouterProvider(apiKey, opts...)`
- `NewMoonshotProvider(apiKey, opts...)`
- `NewPerplexityProvider(apiKey, opts...)`
- `NewQwenProvider(apiKey, opts...)`
- `NewCopilotProvider(githubToken, opts...)`
- `NewLMStudioProvider(opts...)`
- `NewOllamaProvider(opts...)`
- `NewLlamaCPPProvider(baseURL, opts...)`
- `NewAnthropicProvider(apiKey, opts...)`
- `NewGeminiProvider(apiKey, opts...)`
- `NewCloudflareProvider(accountID, apiToken, opts...)`
- `NewOpenAICompatibleProvider(name, baseURL, apiKey, opts...)`

Useful options:

```go
yongle.WithBaseURL("http://localhost:8080/v1")
yongle.WithHTTPClient(customClient)
yongle.WithHeader("HTTP-Referer", "https://example.com")
yongle.WithHeader("X-Title", "My App")
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

This Go port now covers baochuan's core ergonomics, OpenAI-compatible providers including Nous Portal, native adapters for Anthropic, Gemini, and Cloudflare Workers AI, and initial `AgentProvider` support for Hermes Agent. Additional provider-specific features can be layered on the same `Provider` and `AgentProvider` interfaces without changing callers.

## License

MIT
