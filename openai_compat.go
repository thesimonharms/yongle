package yongle

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

type OpenAICompatibleProvider struct {
	providerName string
	baseURL      string
	apiKey       string
	httpClient   *http.Client
	headers      map[string]string
}

func NewOpenAICompatibleProvider(name, baseURL, apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	cfg := applyOptions(baseURL, opts...)
	return &OpenAICompatibleProvider{providerName: name, baseURL: cfg.baseURL, apiKey: apiKey, httpClient: cfg.httpClient, headers: cfg.headers}
}

func (p *OpenAICompatibleProvider) Name() string { return p.providerName }
func (p *OpenAICompatibleProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	req.Stream = false
	var out ChatResponse
	err := doJSONProvider(ctx, p.httpClient, p.providerName, http.MethodPost, p.baseURL+"/chat/completions", p.apiKey, p.headers, req, &out)
	return out, err
}
func (p *OpenAICompatibleProvider) StreamChat(ctx context.Context, req ChatRequest) (ChunkStream, error) {
	req.Stream = true
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if p.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	for k, v := range p.headers {
		hreq.Header.Set(k, v)
	}
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.providerName, StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return streamFromBody(resp.Body), nil
}
func (p *OpenAICompatibleProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	var out modelListResponse
	if err := doJSONProvider(ctx, p.httpClient, p.providerName, http.MethodGet, p.baseURL+"/models", p.apiKey, p.headers, nil, &out); err != nil {
		return nil, err
	}
	return out.Data, nil
}
func (p *OpenAICompatibleProvider) TTS(context.Context, TTSRequest) ([]byte, error) {
	return nil, ErrUnsupported
}

func newOpenAICompatibleDefault(name, defaultBaseURL, apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	cfg := applyOptions(defaultBaseURL, opts...)
	return &OpenAICompatibleProvider{providerName: name, baseURL: cfg.baseURL, apiKey: apiKey, httpClient: cfg.httpClient, headers: cfg.headers}
}

// Provider constructors with current upstream defaults.
func NewOpenAIProvider(apiKey string, opts ...ClientOption) *OpenAIProvider {
	return &OpenAIProvider{OpenAICompatibleProvider: newOpenAICompatibleDefault("openai", "https://api.openai.com/v1", apiKey, opts...)}
}

// NewDeepSeekProvider uses the official OpenAI-compatible base URL
// (https://api.deepseek.com). The /v1 alias also works via WithBaseURL.
func NewDeepSeekProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("deepseek", "https://api.deepseek.com", apiKey, opts...)
}
func NewXAIProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("xai", "https://api.x.ai/v1", apiKey, opts...)
}
func NewMistralProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("mistral", "https://api.mistral.ai/v1", apiKey, opts...)
}
func NewOpenRouterProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("openrouter", "https://openrouter.ai/api/v1", apiKey, opts...)
}

// NewMoonshotProvider targets the global Kimi Open Platform (api.moonshot.ai).
// Use NewMoonshotCNProvider for keys issued on the China platform.
func NewMoonshotProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("moonshot", "https://api.moonshot.ai/v1", apiKey, opts...)
}

// NewMoonshotCNProvider targets the China Moonshot/Kimi platform (api.moonshot.cn).
func NewMoonshotCNProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("moonshot", "https://api.moonshot.cn/v1", apiKey, opts...)
}
func NewPerplexityProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("perplexity", "https://api.perplexity.ai", apiKey, opts...)
}
func NewNousProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("nous", "https://inference-api.nousresearch.com/v1", apiKey, opts...)
}

// NewQwenProvider uses DashScope OpenAI-compatible mode on the international
// endpoint. Use NewQwenCNProvider for Beijing-region keys.
func NewQwenProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", apiKey, opts...)
}

// NewQwenCNProvider uses DashScope OpenAI-compatible mode on the China (Beijing) endpoint.
func NewQwenCNProvider(apiKey string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", apiKey, opts...)
}

// NewCopilotProvider talks to api.githubcopilot.com with the integration id
// required by the Copilot Chat Completions API. Override with WithHeader if needed.
func NewCopilotProvider(githubToken string, opts ...ClientOption) *OpenAICompatibleProvider {
	defaults := []ClientOption{WithHeader("Copilot-Integration-Id", "copilot-developer-cli")}
	return newOpenAICompatibleDefault("copilot", "https://api.githubcopilot.com", githubToken, append(defaults, opts...)...)
}
func NewLMStudioProvider(opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("lmstudio", "http://localhost:1234/v1", "", opts...)
}
func NewOllamaProvider(opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("ollama", "http://localhost:11434/v1", "", opts...)
}
func NewLlamaCPPProvider(baseURL string, opts ...ClientOption) *OpenAICompatibleProvider {
	return newOpenAICompatibleDefault("llamacpp", baseURL, "", opts...)
}
