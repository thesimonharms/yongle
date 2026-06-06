package yongle

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNousPortalProviderUsesOpenAICompatibleInferenceAPI(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-nous","model":"Hermes-4-405B","choices":[{"index":0,"message":{"role":"assistant","content":"from Nous Portal"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4}}`))
	}))
	defer srv.Close()

	provider := NewNousProvider("nous-secret", WithBaseURL(srv.URL))
	if provider.Name() != "nous" {
		t.Fatalf("provider name = %q", provider.Name())
	}
	resp, err := provider.Chat(context.Background(), mustReq(t, NewChatRequest("Hermes-4-405B").User("hello")))
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/chat/completions" || gotAuth != "Bearer nous-secret" {
		t.Fatalf("bad request path/auth: %s %s", gotPath, gotAuth)
	}
	if gotBody["model"] != "Hermes-4-405B" {
		t.Fatalf("bad model in body: %#v", gotBody)
	}
	if resp.Content() != "from Nous Portal" || resp.Usage.TotalTokens != 4 {
		t.Fatalf("bad response: %#v", resp)
	}
}
