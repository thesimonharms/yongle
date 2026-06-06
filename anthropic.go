package yongle

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const anthropicVersion = "2023-06-01"
const anthropicDefaultMaxTokens uint32 = 4096

type AnthropicProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

func NewAnthropicProvider(apiKey string, opts ...ClientOption) *AnthropicProvider {
	cfg := applyOptions("https://api.anthropic.com/v1", opts...)
	return &AnthropicProvider{apiKey: apiKey, baseURL: cfg.baseURL, httpClient: cfg.httpClient, headers: cfg.headers}
}

func (p *AnthropicProvider) Name() string { return "anthropic" }
func (p *AnthropicProvider) TTS(context.Context, TTSRequest) ([]byte, error) {
	return nil, ErrUnsupported
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   uint32             `json:"max_tokens"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	Stream      bool               `json:"stream"`
	Temperature *float32           `json:"temperature,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	ToolChoice  any                `json:"tool_choice,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicContentBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Source    any    `json:"source,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

type anthropicSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

func anthropicHeaders(p *AnthropicProvider, req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	for k, v := range p.headers {
		req.Header.Set(k, v)
	}
}

func toAnthropicRequest(req ChatRequest, stream bool) anthropicRequest {
	var system []string
	var messages []anthropicMessage
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			system = append(system, m.Content.Text())
			continue
		}
		messages = append(messages, toAnthropicMessage(m))
	}
	maxTokens := anthropicDefaultMaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	out := anthropicRequest{Model: req.Model, MaxTokens: maxTokens, Messages: messages, System: strings.Join(system, "\n"), Stream: stream, Temperature: req.Temperature}
	for _, t := range req.Tools {
		if t.Function == nil {
			continue
		}
		params := t.Function.Parameters
		if params == nil {
			params = map[string]any{"type": "object"}
		}
		out.Tools = append(out.Tools, anthropicTool{Name: t.Function.Name, Description: t.Function.Description, InputSchema: params})
	}
	if req.ToolChoice != nil {
		out.ToolChoice = toAnthropicToolChoice(*req.ToolChoice)
	}
	return out
}

func toAnthropicMessage(m ChatMessage) anthropicMessage {
	if m.Role == RoleTool {
		return anthropicMessage{Role: "user", Content: []anthropicContentBlock{{Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content.Text()}}}
	}
	role := "user"
	if m.Role == RoleAssistant {
		role = "assistant"
	}
	blocks := contentToAnthropicBlocks(m.Content)
	for _, tc := range m.ToolCalls {
		if tc.Function == nil {
			continue
		}
		var input any = map[string]any{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
		blocks = append(blocks, anthropicContentBlock{Type: "tool_use", ID: tc.ID, Name: tc.Function.Name, Input: input})
	}
	if len(m.ToolCalls) == 0 && m.Content.text != nil {
		return anthropicMessage{Role: role, Content: m.Content.Text()}
	}
	return anthropicMessage{Role: role, Content: blocks}
}

func contentToAnthropicBlocks(c MessageContent) []anthropicContentBlock {
	if c.text != nil {
		return []anthropicContentBlock{{Type: "text", Text: c.Text()}}
	}
	blocks := make([]anthropicContentBlock, 0, len(c.parts))
	for _, p := range c.parts {
		switch p.Type {
		case "text":
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: p.Text})
		case "image_url":
			if p.ImageURL == nil {
				continue
			}
			if mt, data, ok := parseDataURL(p.ImageURL.URL); ok {
				blocks = append(blocks, anthropicContentBlock{Type: "image", Source: anthropicSource{Type: "base64", MediaType: mt, Data: data}})
			} else {
				blocks = append(blocks, anthropicContentBlock{Type: "image", Source: anthropicSource{Type: "url", URL: p.ImageURL.URL}})
			}
		case "document":
			if p.Document != nil {
				blocks = append(blocks, anthropicContentBlock{Type: "document", Source: anthropicSource{Type: "base64", MediaType: p.Document.MediaType, Data: p.Document.Data}})
			}
		}
	}
	return blocks
}

func toAnthropicToolChoice(choice ToolChoice) any {
	b, _ := json.Marshal(choice)
	var v any
	_ = json.Unmarshal(b, &v)
	switch x := v.(type) {
	case string:
		if x == "required" {
			return map[string]any{"type": "any"}
		}
		return map[string]any{"type": x}
	case map[string]any:
		if fn, ok := x["function"].(map[string]any); ok {
			return map[string]any{"type": "tool", "name": fn["name"]}
		}
	}
	return nil
}

func (p *AnthropicProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	body := toAnthropicRequest(req, false)
	var wire anthropicResponse
	if err := p.doAnthropicJSON(ctx, http.MethodPost, p.baseURL+"/messages", body, &wire); err != nil {
		return ChatResponse{}, err
	}
	return fromAnthropicResponse(wire), nil
}

func (p *AnthropicProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	var wire struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
		} `json:"data"`
	}
	if err := p.doAnthropicJSON(ctx, http.MethodGet, p.baseURL+"/models", nil, &wire); err != nil {
		return nil, err
	}
	models := make([]ModelInfo, 0, len(wire.Data))
	for _, m := range wire.Data {
		models = append(models, ModelInfo{ID: m.ID, OwnedBy: "anthropic", Object: "model"})
	}
	return models, nil
}

func (p *AnthropicProvider) doAnthropicJSON(ctx context.Context, method, url string, body any, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return err
	}
	anthropicHeaders(p, req)
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(b))}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type anthropicResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Content []struct {
		Type  string `json:"type"`
		Text  string `json:"text"`
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`
	} `json:"content"`
	StopReason *string `json:"stop_reason"`
	Usage      struct {
		InputTokens  uint32 `json:"input_tokens"`
		OutputTokens uint32 `json:"output_tokens"`
	} `json:"usage"`
}

func fromAnthropicResponse(r anthropicResponse) ChatResponse {
	var text string
	var calls []ToolCall
	for _, b := range r.Content {
		if b.Type == "text" {
			text += b.Text
		}
		if b.Type == "tool_use" {
			args, _ := json.Marshal(b.Input)
			calls = append(calls, ToolCall{ID: b.ID, Type: "function", Function: &FunctionCall{Name: b.Name, Arguments: string(args)}})
		}
	}
	msg := AssistantMessage(text)
	msg.ToolCalls = calls
	return ChatResponse{ID: r.ID, Model: r.Model, Choices: []ChatChoice{{Index: 0, Message: msg, FinishReason: r.StopReason}}, Usage: &Usage{PromptTokens: r.Usage.InputTokens, CompletionTokens: r.Usage.OutputTokens, TotalTokens: r.Usage.InputTokens + r.Usage.OutputTokens}}
}

func (p *AnthropicProvider) StreamChat(ctx context.Context, req ChatRequest) (ChunkStream, error) {
	body := toAnthropicRequest(req, true)
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	anthropicHeaders(p, hreq)
	hreq.Header.Set("Accept", "text/event-stream")
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return anthropicStreamFromBody(resp.Body, req.Model), nil
}

func anthropicStreamFromBody(body io.ReadCloser, model string) ChunkStream {
	return func(yield func(StreamChunk, error) bool) {
		defer body.Close()
		s := bufio.NewScanner(body)
		var data strings.Builder
		idx := 0
		emit := func(payload string) bool {
			var ev struct {
				Message struct {
					ID    string `json:"id"`
					Model string `json:"model"`
				} `json:"message"`
				Delta struct {
					Text string `json:"text"`
					Type string `json:"type"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				return yield(StreamChunk{}, err)
			}
			if ev.Delta.Text == "" {
				return true
			}
			idx++
			return yield(StreamChunk{ID: fmt.Sprintf("anthropic-chunk-%d", idx), Model: model, Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: &ev.Delta.Text}}}}, nil)
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
