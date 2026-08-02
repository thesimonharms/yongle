package yongle

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatRequestBuilderBuildsAndValidates(t *testing.T) {
	_, err := NewChatRequest("deepseek-chat").Build()
	if err == nil || !strings.Contains(err.Error(), "at least one message") {
		t.Fatalf("expected missing-message validation, got %v", err)
	}

	req, err := NewChatRequest("deepseek-chat").
		System("You are concise.").
		User("Tell me about Zheng He's treasure ships.").
		MaxTokens(512).
		MaxCompletionTokens(256).
		Temperature(0.2).
		TopP(0.9).
		Tool(Tool{Type: "function", Function: &FunctionDefinition{Name: "lookup", Description: "lookup things", Parameters: map[string]any{"type": "object"}}}).
		ToolChoice(ToolChoiceAuto()).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "deepseek-chat" || len(req.Messages) != 2 || req.MaxTokens == nil || *req.MaxTokens != 512 {
		t.Fatalf("unexpected request: %#v", req)
	}
	if req.MaxCompletionTokens == nil || *req.MaxCompletionTokens != 256 {
		t.Fatalf("expected max_completion_tokens, got %#v", req.MaxCompletionTokens)
	}
	if req.Messages[1].Content.Text() != "Tell me about Zheng He's treasure ships." {
		t.Fatalf("unexpected content: %#v", req.Messages[1].Content)
	}
}

func TestProviderDefaultsMatchCurrentAPIs(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		header  string
		value   string
		p       *OpenAICompatibleProvider
	}{
		{"deepseek", "https://api.deepseek.com", "", "", NewDeepSeekProvider("k")},
		{"moonshot", "https://api.moonshot.ai/v1", "", "", NewMoonshotProvider("k")},
		{"moonshot-cn", "https://api.moonshot.cn/v1", "", "", NewMoonshotCNProvider("k")},
		{"qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "", "", NewQwenProvider("k")},
		{"qwen-cn", "https://dashscope.aliyuncs.com/compatible-mode/v1", "", "", NewQwenCNProvider("k")},
		{"copilot", "https://api.githubcopilot.com", "Copilot-Integration-Id", "copilot-developer-cli", NewCopilotProvider("gh")},
		{"openai", "https://api.openai.com/v1", "", "", NewOpenAIProvider("k").OpenAICompatibleProvider},
		{"xai", "https://api.x.ai/v1", "", "", NewXAIProvider("k")},
		{"mistral", "https://api.mistral.ai/v1", "", "", NewMistralProvider("k")},
		{"openrouter", "https://openrouter.ai/api/v1", "", "", NewOpenRouterProvider("k")},
		{"perplexity", "https://api.perplexity.ai", "", "", NewPerplexityProvider("k")},
		{"nous", "https://inference-api.nousresearch.com/v1", "", "", NewNousProvider("k")},
		{"ollama", "http://localhost:11434/v1", "", "", NewOllamaProvider()},
		{"lmstudio", "http://localhost:1234/v1", "", "", NewLMStudioProvider()},
	}
	for _, tc := range cases {
		if tc.p.baseURL != tc.baseURL {
			t.Fatalf("%s base URL: got %q want %q", tc.name, tc.p.baseURL, tc.baseURL)
		}
		if tc.header != "" && tc.p.headers[tc.header] != tc.value {
			t.Fatalf("%s header %s: got %q want %q", tc.name, tc.header, tc.p.headers[tc.header], tc.value)
		}
	}
	if NewAnthropicProvider("k").baseURL != "https://api.anthropic.com/v1" {
		t.Fatal("anthropic base URL drift")
	}
	if NewGeminiProvider("k").baseURL != "https://generativelanguage.googleapis.com/v1beta" {
		t.Fatal("gemini base URL drift")
	}
	if NewCloudflareProvider("acct", "tok").baseURL != "https://api.cloudflare.com/client/v4" {
		t.Fatal("cloudflare base URL drift")
	}
}

func TestOpenAICompatibleChatSendsExpectedWireRequestAndParsesResponse(t *testing.T) {
	var gotAuth, gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl-1","model":"deepseek-chat","choices":[{"index":0,"message":{"role":"assistant","content":"A treasure fleet."},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`))
	}))
	defer srv.Close()

	provider := NewOpenAICompatibleProvider("deepseek", srv.URL, "secret")
	resp, err := provider.Chat(context.Background(), mustReq(t, NewChatRequest("deepseek-chat").User("hi")))
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/chat/completions" || gotAuth != "Bearer secret" {
		t.Fatalf("wrong request path/auth: %s %s", gotPath, gotAuth)
	}
	if gotBody["model"] != "deepseek-chat" || gotBody["stream"] != false {
		t.Fatalf("wrong request body: %#v", gotBody)
	}
	if resp.Content() != "A treasure fleet." || resp.Usage == nil || resp.Usage.TotalTokens != 7 {
		t.Fatalf("unexpected response: %#v", resp)
	}
}

func TestOpenAICompatibleStreamParsesSSEChunks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("no flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Bao\"},\"finish_reason\":null}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"chuan\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	provider := NewOpenAICompatibleProvider("test", srv.URL, "secret")
	stream, err := provider.StreamChat(context.Background(), mustReq(t, NewChatRequest("m").User("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for chunk, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString(chunk.DeltaContent())
	}
	if b.String() != "Baochuan" {
		t.Fatalf("unexpected stream text %q", b.String())
	}
}

func TestModelsAndTTS(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o","object":"model","owned_by":"openai"}]}`))
		case "/audio/speech":
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = w.Write([]byte("mp3-bytes"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	provider := NewOpenAIProvider("secret", WithBaseURL(srv.URL))
	models, err := provider.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "gpt-4o" {
		t.Fatalf("models = %#v, err=%v", models, err)
	}
	audio, err := provider.TTS(context.Background(), mustTTS(t, NewTTSRequest("tts-1", "hello").Voice("nova").Format("mp3")))
	if err != nil || string(audio) != "mp3-bytes" {
		t.Fatalf("tts = %q, err=%v", string(audio), err)
	}
}

func TestSSEScannerIgnoresCommentsEventsAndDone(t *testing.T) {
	input := strings.NewReader(": ping\nevent: message\ndata: {\"id\":\"1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
	chunks, errs := parseSSE(bufio.NewReader(input))
	var count int
	for chunk := range chunks {
		count++
		if chunk.DeltaContent() != "x" {
			t.Fatalf("wrong delta: %#v", chunk)
		}
	}
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("got %d chunks", count)
	}
}

func mustReq(t *testing.T, b *ChatRequestBuilder) ChatRequest {
	t.Helper()
	req, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func mustTTS(t *testing.T, b *TTSRequestBuilder) TTSRequest {
	t.Helper()
	req, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	return req
}
