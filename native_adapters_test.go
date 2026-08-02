package yongle

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAnthropicNativeChatModelsAndStream(t *testing.T) {
	var chatPath, authHeader, versionHeader string
	var chatBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/messages":
			chatPath = r.URL.Path
			authHeader = r.Header.Get("x-api-key")
			versionHeader = r.Header.Get("anthropic-version")
			if err := json.NewDecoder(r.Body).Decode(&chatBody); err != nil {
				t.Fatal(err)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-3-5-sonnet","content":[{"type":"text","text":"native Claude"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":7}}`))
		case "/models":
			if r.Header.Get("x-api-key") != "anthropic-secret" {
				t.Fatalf("missing anthropic auth")
			}
			_, _ = w.Write([]byte(`{"data":[{"id":"claude-3-5-sonnet","display_name":"Claude 3.5 Sonnet"}]}`))
		case "/stream/messages":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n"))
			_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	provider := NewAnthropicProvider("anthropic-secret", WithBaseURL(srv.URL))
	resp, err := provider.Chat(context.Background(), mustReq(t, NewChatRequest("claude-3-5-sonnet").System("be brief").User("hi").MaxTokens(123)))
	if err != nil {
		t.Fatal(err)
	}
	if chatPath != "/messages" || authHeader != "anthropic-secret" || versionHeader == "" {
		t.Fatalf("bad anthropic request headers/path")
	}
	if chatBody["system"] != "be brief" || chatBody["max_tokens"].(float64) != 123 || chatBody["stream"] != false {
		t.Fatalf("bad anthropic body: %#v", chatBody)
	}
	messages := chatBody["messages"].([]any)
	if len(messages) != 1 || messages[0].(map[string]any)["role"] != "user" {
		t.Fatalf("system should be top-level, got %#v", messages)
	}
	if resp.Content() != "native Claude" || resp.Usage.TotalTokens != 12 {
		t.Fatalf("bad anthropic response: %#v", resp)
	}

	models, err := provider.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "claude-3-5-sonnet" || models[0].OwnedBy != "anthropic" {
		t.Fatalf("models %#v err %v", models, err)
	}

	streamProvider := NewAnthropicProvider("anthropic-secret", WithBaseURL(srv.URL+"/stream"))
	stream, err := streamProvider.StreamChat(context.Background(), mustReq(t, NewChatRequest("claude-3-5-sonnet").User("hi")))
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
	if b.String() != "hello" {
		t.Fatalf("bad stream %q", b.String())
	}
}

func TestGeminiNativeChatModelsAndStream(t *testing.T) {
	var chatPath, chatQuery, chatAuth string
	var chatBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/models/gemini-2.5-flash:generateContent":
			chatPath = r.URL.Path
			chatQuery = r.URL.RawQuery
			chatAuth = r.Header.Get("x-goog-api-key")
			if err := json.NewDecoder(r.Body).Decode(&chatBody); err != nil {
				t.Fatal(err)
			}
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"native Gemini"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}`))
		case r.URL.Path == "/models":
			if r.Header.Get("x-goog-api-key") != "gemini-key" {
				t.Fatalf("missing gemini auth header")
			}
			_, _ = w.Write([]byte(`{"models":[{"name":"models/gemini-2.5-flash","displayName":"Gemini Flash","inputTokenLimit":1048576}]}`))
		case r.URL.Path == "/models/gemini-2.5-flash:streamGenerateContent":
			if r.URL.Query().Get("alt") != "sse" {
				t.Fatalf("missing alt=sse")
			}
			if r.Header.Get("x-goog-api-key") != "gemini-key" {
				t.Fatalf("missing gemini auth header on stream")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Gem\"}]},\"index\":0}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ini\"}]},\"index\":0,\"finishReason\":\"STOP\"}]}\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	provider := NewGeminiProvider("gemini-key", WithBaseURL(srv.URL))
	resp, err := provider.Chat(context.Background(), mustReq(t, NewChatRequest("gemini-2.5-flash").System("be kind").User("hi").MaxTokens(55).Temperature(0.1)))
	if err != nil {
		t.Fatal(err)
	}
	if chatPath == "" || chatAuth != "gemini-key" || strings.Contains(chatQuery, "key=") {
		t.Fatalf("bad gemini auth/URL: path=%s auth=%q query=%q", chatPath, chatAuth, chatQuery)
	}
	if _, ok := chatBody["systemInstruction"]; !ok {
		t.Fatalf("missing systemInstruction in %#v", chatBody)
	}
	gen := chatBody["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 55 {
		t.Fatalf("bad generation config %#v", gen)
	}
	if resp.Content() != "native Gemini" || resp.Usage.TotalTokens != 5 {
		t.Fatalf("bad gemini response %#v", resp)
	}

	models, err := provider.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "gemini-2.5-flash" || models[0].OwnedBy != "google" {
		t.Fatalf("models %#v err %v", models, err)
	}
	if models[0].ContextLength == nil || *models[0].ContextLength != 1048576 || models[0].DisplayName != "Gemini Flash" {
		t.Fatalf("expected context/display metadata, got %#v", models[0])
	}

	stream, err := provider.StreamChat(context.Background(), mustReq(t, NewChatRequest("gemini-2.5-flash").User("hi")))
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
	if b.String() != "Gemini" {
		t.Fatalf("bad stream %q", b.String())
	}
}

func TestCloudflareNativeChatModelsAndStream(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/accounts/acct/ai/run/@cf/meta/llama-3.1-8b-instruct":
			gotPath = r.URL.Path
			gotAuth = r.Header.Get("Authorization")
			if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
				t.Fatal(err)
			}
			if gotBody["stream"] == true {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"response\":\"Cloud\"}\n\n"))
				_, _ = w.Write([]byte("data: {\"response\":\"flare\"}\n\n"))
				return
			}
			_, _ = w.Write([]byte(`{"success":true,"result":{"response":"native Cloudflare"},"errors":[]}`))
		case "/accounts/acct/ai/models/search":
			if r.URL.Query().Get("task") != "Text Generation" {
				t.Fatalf("missing model filter")
			}
			_, _ = w.Write([]byte(`{"result":[{"id":"@cf/meta/llama-3.1-8b-instruct","description":"Llama","task":{"name":"Text Generation"}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	provider := NewCloudflareProvider("acct", "cf-token", WithBaseURL(srv.URL))
	req := mustReq(t, NewChatRequest("@cf/meta/llama-3.1-8b-instruct").System("s").User("u").MaxTokens(44))
	resp, err := provider.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if gotPath == "" || gotAuth != "Bearer cf-token" {
		t.Fatalf("bad cloudflare request")
	}
	if gotBody["max_tokens"].(float64) != 44 || gotBody["stream"] != false {
		t.Fatalf("bad cloudflare body %#v", gotBody)
	}
	if resp.Content() != "native Cloudflare" {
		t.Fatalf("bad response %#v", resp)
	}

	models, err := provider.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "@cf/meta/llama-3.1-8b-instruct" {
		t.Fatalf("models %#v err %v", models, err)
	}

	stream, err := provider.StreamChat(context.Background(), req)
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
	if b.String() != "Cloudflare" {
		t.Fatalf("bad stream %q", b.String())
	}
}
