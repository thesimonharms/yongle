package yongle

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAgentRunRequestBuilderBuildsAndValidates(t *testing.T) {
	_, err := NewAgentRunRequest("hermes-agent", "").Build()
	if err == nil || !strings.Contains(err.Error(), "input") {
		t.Fatalf("expected missing-input validation, got %v", err)
	}

	req, err := NewAgentRunRequest("hermes-agent", "What files are here?").
		Instructions("You are a coding assistant.").
		PreviousResponseID("resp_previous").
		Conversation("yongle").
		Store(true).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "hermes-agent" || req.Input.Text != "What files are here?" {
		t.Fatalf("unexpected request: %#v", req)
	}
	if req.Instructions == nil || *req.Instructions != "You are a coding assistant." {
		t.Fatalf("unexpected instructions: %#v", req.Instructions)
	}
	if req.PreviousResponseID == nil || *req.PreviousResponseID != "resp_previous" {
		t.Fatalf("unexpected previous response id: %#v", req.PreviousResponseID)
	}
	if req.Conversation == nil || *req.Conversation != "yongle" {
		t.Fatalf("unexpected conversation: %#v", req.Conversation)
	}
	if req.Store == nil || *req.Store != true {
		t.Fatalf("unexpected store: %#v", req.Store)
	}
	if req.Stream {
		t.Fatalf("builder should default stream to false")
	}
}

func TestHermesAgentRunUsesResponsesAPI(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_abc123","object":"response","status":"completed","model":"hermes-agent","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"README.md *.go"}]}],"usage":{"input_tokens":50,"output_tokens":12,"total_tokens":62}}`))
	}))
	defer srv.Close()

	provider := NewHermesAgentProvider("test-api-key", WithBaseURL(srv.URL+"/v1"))
	resp, err := provider.Run(context.Background(), mustAgentReq(t, NewAgentRunRequest("hermes-agent", "What files are here?").PreviousResponseID("resp_previous").Store(true)))
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/responses" || gotAuth != "Bearer test-api-key" {
		t.Fatalf("wrong request path/auth: %s %s", gotPath, gotAuth)
	}
	if gotBody["model"] != "hermes-agent" || gotBody["previous_response_id"] != "resp_previous" || gotBody["input"] != "What files are here?" {
		t.Fatalf("wrong request body: %#v", gotBody)
	}
	if resp.ID != "resp_abc123" || resp.Status != "completed" || resp.OutputText() != "README.md *.go" {
		t.Fatalf("unexpected response: %#v", resp)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 62 {
		t.Fatalf("unexpected usage: %#v", resp.Usage)
	}
}

func TestHermesAgentStreamRunYieldsResponseEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["stream"] != true {
			t.Fatalf("expected stream=true body, got %#v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			"event: response.created\n",
			"data: {\"id\":\"resp_abc123\",\"type\":\"response.created\"}\n",
			"\n",
			"event: hermes.tool.progress\n",
			"data: {\"tool\":\"terminal\",\"status\":\"started\"}\n",
			"\n",
			"event: response.output_text.delta\n",
			"data: {\"delta\":\"README.md\"}\n",
			"\n",
			"event: response.completed\n",
			"data: {\"response\":{\"id\":\"resp_abc123\",\"status\":\"completed\"}}\n",
			"\n",
			"data: [DONE]\n",
		}, "")))
	}))
	defer srv.Close()

	provider := NewHermesAgentProvider("test-api-key", WithBaseURL(srv.URL+"/v1"))
	stream, err := provider.StreamRun(context.Background(), mustAgentReq(t, NewAgentRunRequest("hermes-agent", "List files")))
	if err != nil {
		t.Fatal(err)
	}

	var eventNames []string
	var textDelta strings.Builder
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event.Event != "" {
			eventNames = append(eventNames, event.Event)
		}
		textDelta.WriteString(event.OutputTextDelta())
	}
	want := []string{"response.created", "hermes.tool.progress", "response.output_text.delta", "response.completed"}
	if strings.Join(eventNames, ",") != strings.Join(want, ",") {
		t.Fatalf("event names = %#v, want %#v", eventNames, want)
	}
	if textDelta.String() != "README.md" {
		t.Fatalf("text delta = %q", textDelta.String())
	}
}

func TestAgentSSEStopsAtDoneWithoutTrailingBlankLine(t *testing.T) {
	stream := agentEventsFromBody(io.NopCloser(strings.NewReader("event: response.output_text.delta\ndata: {\"delta\":\"done\"}\n\ndata: [DONE]")))

	var count int
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if event.OutputTextDelta() != "done" {
			t.Fatalf("unexpected delta: %#v", event)
		}
	}
	if count != 1 {
		t.Fatalf("got %d events", count)
	}
}

func TestHermesAgentProviderName(t *testing.T) {
	if NewHermesAgentProvider("key").Name() != "hermes-agent" {
		t.Fatal("unexpected provider name")
	}
}

func mustAgentReq(t *testing.T, b *AgentRunRequestBuilder) AgentRunRequest {
	t.Helper()
	req, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	return req
}
