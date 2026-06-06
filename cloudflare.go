package yongle

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

type CloudflareProvider struct {
	accountID, apiToken, baseURL string
	httpClient                   *http.Client
	headers                      map[string]string
}

func NewCloudflareProvider(accountID, apiToken string, opts ...ClientOption) *CloudflareProvider {
	cfg := applyOptions("https://api.cloudflare.com/client/v4", opts...)
	return &CloudflareProvider{accountID: accountID, apiToken: apiToken, baseURL: cfg.baseURL, httpClient: cfg.httpClient, headers: cfg.headers}
}
func (p *CloudflareProvider) Name() string { return "cloudflare" }
func (p *CloudflareProvider) TTS(context.Context, TTSRequest) ([]byte, error) {
	return nil, ErrUnsupported
}

type cfChatRequest struct {
	Messages    []cfMessage `json:"messages"`
	Stream      bool        `json:"stream"`
	MaxTokens   *uint32     `json:"max_tokens,omitempty"`
	Temperature *float32    `json:"temperature,omitempty"`
}
type cfMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type cfEnvelope struct {
	Success bool          `json:"success"`
	Result  *cfChatResult `json:"result"`
	Errors  []struct {
		Message string `json:"message"`
	} `json:"errors"`
}
type cfChatResult struct {
	Response string `json:"response"`
}

func (p *CloudflareProvider) runURL(model string) string {
	return fmt.Sprintf("%s/accounts/%s/ai/run/%s", p.baseURL, p.accountID, model)
}
func (p *CloudflareProvider) modelsURL() string {
	return fmt.Sprintf("%s/accounts/%s/ai/models/search", p.baseURL, p.accountID)
}

func toCFRequest(req ChatRequest, stream bool) cfChatRequest {
	msgs := make([]cfMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, cfMessage{Role: string(m.Role), Content: m.Content.Text()})
	}
	return cfChatRequest{Messages: msgs, Stream: stream, MaxTokens: req.MaxTokens, Temperature: req.Temperature}
}

func (p *CloudflareProvider) addCFHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+p.apiToken)
	for k, v := range p.headers {
		req.Header.Set(k, v)
	}
}

func (p *CloudflareProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	b, err := json.Marshal(toCFRequest(req, false))
	if err != nil {
		return ChatResponse{}, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.runURL(req.Model), bytes.NewReader(b))
	if err != nil {
		return ChatResponse{}, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	p.addCFHeaders(hreq)
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return ChatResponse{}, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	var env cfEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		return ChatResponse{}, err
	}
	if !env.Success {
		parts := make([]string, 0, len(env.Errors))
		for _, e := range env.Errors {
			parts = append(parts, e.Message)
		}
		return ChatResponse{}, &Error{Provider: p.Name(), StatusCode: 200, Message: strings.Join(parts, "; ")}
	}
	if env.Result == nil {
		return ChatResponse{}, &Error{Provider: p.Name(), StatusCode: 200, Message: "empty result from Cloudflare"}
	}
	return ChatResponse{Model: req.Model, Choices: []ChatChoice{{Index: 0, Message: AssistantMessage(env.Result.Response), FinishReason: strPtr("stop")}}}, nil
}

func (p *CloudflareProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	u, _ := url.Parse(p.modelsURL())
	q := u.Query()
	q.Set("task", "Text Generation")
	u.RawQuery = q.Encode()
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	p.addCFHeaders(hreq)
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	var wire struct {
		Result []struct {
			ID          string `json:"id"`
			Description string `json:"description"`
			Task        *struct {
				Name string `json:"name"`
			} `json:"task"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return nil, err
	}
	out := make([]ModelInfo, 0, len(wire.Result))
	for _, m := range wire.Result {
		owner := ""
		if m.Task != nil {
			owner = m.Task.Name
		}
		out = append(out, ModelInfo{ID: m.ID, OwnedBy: owner, Object: "model"})
	}
	return out, nil
}

func (p *CloudflareProvider) StreamChat(ctx context.Context, req ChatRequest) (ChunkStream, error) {
	b, err := json.Marshal(toCFRequest(req, true))
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.runURL(req.Model), bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	p.addCFHeaders(hreq)
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return cfStreamFromBody(resp.Body, req.Model), nil
}

func cfStreamFromBody(body io.ReadCloser, model string) ChunkStream {
	return func(yield func(StreamChunk, error) bool) {
		defer body.Close()
		s := bufio.NewScanner(body)
		var data strings.Builder
		idx := 0
		emit := func(payload string) bool {
			if payload == "[DONE]" {
				return true
			}
			var r struct {
				Response string `json:"response"`
				Error    string `json:"error"`
			}
			if err := json.Unmarshal([]byte(payload), &r); err != nil {
				return yield(StreamChunk{}, err)
			}
			if r.Error != "" {
				return yield(StreamChunk{}, &Error{Provider: "cloudflare", Message: r.Error})
			}
			if r.Response == "" {
				return true
			}
			idx++
			text := r.Response
			return yield(StreamChunk{ID: fmt.Sprintf("cloudflare-chunk-%d", idx), Model: model, Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: &text}}}}, nil)
		}
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			if line == "" {
				if data.Len() > 0 {
					if !emit(data.String()) {
						return
					}
					data.Reset()
				}
				continue
			}
			if strings.HasPrefix(line, "data:") {
				if data.Len() > 0 {
					data.WriteByte('\n')
				}
				data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
		if err := s.Err(); err != nil {
			yield(StreamChunk{}, err)
		}
	}
}

func strPtr(s string) *string { return &s }
