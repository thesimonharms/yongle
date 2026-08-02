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

type GeminiProvider struct {
	apiKey, baseURL string
	httpClient      *http.Client
	headers         map[string]string
}

func NewGeminiProvider(apiKey string, opts ...ClientOption) *GeminiProvider {
	cfg := applyOptions("https://generativelanguage.googleapis.com/v1beta", opts...)
	return &GeminiProvider{apiKey: apiKey, baseURL: cfg.baseURL, httpClient: cfg.httpClient, headers: cfg.headers}
}
func (p *GeminiProvider) Name() string                                    { return "gemini" }
func (p *GeminiProvider) TTS(context.Context, TTSRequest) ([]byte, error) { return nil, ErrUnsupported }

type geminiRequest struct {
	Contents          []geminiContent          `json:"contents"`
	SystemInstruction *geminiSystemInstruction `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig  `json:"generationConfig,omitempty"`
	Tools             []geminiTools            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig        `json:"toolConfig,omitempty"`
}
type geminiSystemInstruction struct {
	Parts []geminiPart `json:"parts"`
}
type geminiGenerationConfig struct {
	MaxOutputTokens *uint32  `json:"maxOutputTokens,omitempty"`
	Temperature     *float32 `json:"temperature,omitempty"`
}
type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}
type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}
type geminiInlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}
type geminiFileData struct {
	MIMEType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}
type geminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}
type geminiFunctionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}
type geminiTools struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}
type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}
type geminiToolConfig struct {
	FunctionCallingConfig geminiFunctionCallingConfig `json:"functionCallingConfig"`
}
type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

func (p *GeminiProvider) geminiURL(model, suffix string, stream bool) string {
	u := fmt.Sprintf("%s/models/%s:%s", p.baseURL, model, suffix)
	if stream {
		return u + "?alt=sse"
	}
	return u
}

func (p *GeminiProvider) addGeminiAuth(req *http.Request) {
	if p.apiKey != "" {
		req.Header.Set("x-goog-api-key", p.apiKey)
	}
	for k, v := range p.headers {
		req.Header.Set(k, v)
	}
}

func toGeminiRequest(req ChatRequest) geminiRequest {
	var system []geminiPart
	var contents []geminiContent
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			system = append(system, geminiPart{Text: m.Content.Text()})
			continue
		}
		contents = append(contents, toGeminiContent(m))
	}
	out := geminiRequest{Contents: contents}
	if len(system) > 0 {
		out.SystemInstruction = &geminiSystemInstruction{Parts: system}
	}
	if req.MaxTokens != nil || req.Temperature != nil {
		out.GenerationConfig = &geminiGenerationConfig{MaxOutputTokens: req.MaxTokens, Temperature: req.Temperature}
	}
	if len(req.Tools) > 0 {
		var decls []geminiFunctionDeclaration
		for _, t := range req.Tools {
			if t.Function == nil {
				continue
			}
			decls = append(decls, geminiFunctionDeclaration{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
		}
		out.Tools = []geminiTools{{FunctionDeclarations: decls}}
	}
	if req.ToolChoice != nil {
		out.ToolConfig = toGeminiToolConfig(*req.ToolChoice)
	}
	return out
}

func toGeminiContent(m ChatMessage) geminiContent {
	if m.Role == RoleTool {
		return geminiContent{Role: "user", Parts: []geminiPart{{FunctionResponse: &geminiFunctionResponse{Name: m.ToolCallID, Response: jsonObjectFromString(m.Content.Text())}}}}
	}
	role := "user"
	if m.Role == RoleAssistant {
		role = "model"
	}
	parts := contentToGeminiParts(m.Content)
	for _, tc := range m.ToolCalls {
		if tc.Function == nil {
			continue
		}
		var args any = map[string]any{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{Name: tc.Function.Name, Args: args}})
	}
	return geminiContent{Role: role, Parts: parts}
}

func contentToGeminiParts(c MessageContent) []geminiPart {
	if c.text != nil {
		return []geminiPart{{Text: c.Text()}}
	}
	parts := make([]geminiPart, 0, len(c.parts))
	for _, p := range c.parts {
		switch p.Type {
		case "text":
			parts = append(parts, geminiPart{Text: p.Text})
		case "image_url":
			if p.ImageURL != nil {
				if mt, data, ok := parseDataURL(p.ImageURL.URL); ok {
					parts = append(parts, geminiPart{InlineData: &geminiInlineData{MIMEType: mt, Data: data}})
				} else {
					parts = append(parts, geminiPart{FileData: &geminiFileData{MIMEType: guessImageMIMEType(p.ImageURL.URL), FileURI: p.ImageURL.URL}})
				}
			}
		case "input_audio":
			if p.InputAudio != nil {
				parts = append(parts, geminiPart{InlineData: &geminiInlineData{MIMEType: p.InputAudio.MIMEType(), Data: p.InputAudio.Data}})
			}
		case "document":
			if p.Document != nil {
				parts = append(parts, geminiPart{InlineData: &geminiInlineData{MIMEType: p.Document.MediaType, Data: p.Document.Data}})
			}
		}
	}
	return parts
}

func toGeminiToolConfig(choice ToolChoice) *geminiToolConfig {
	b, _ := json.Marshal(choice)
	var v any
	_ = json.Unmarshal(b, &v)
	cfg := geminiFunctionCallingConfig{Mode: "AUTO"}
	switch x := v.(type) {
	case string:
		switch x {
		case "required":
			cfg.Mode = "ANY"
		case "none":
			cfg.Mode = "NONE"
		default:
			cfg.Mode = "AUTO"
		}
	case map[string]any:
		cfg.Mode = "ANY"
		if fn, ok := x["function"].(map[string]any); ok {
			if name, _ := fn["name"].(string); name != "" {
				cfg.AllowedFunctionNames = []string{name}
			}
		}
	}
	return &geminiToolConfig{FunctionCallingConfig: cfg}
}

func (p *GeminiProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	var wire geminiResponse
	if err := p.doGeminiJSON(ctx, http.MethodPost, p.geminiURL(req.Model, "generateContent", false), toGeminiRequest(req), &wire); err != nil {
		return ChatResponse{}, err
	}
	return fromGeminiResponse(wire, req.Model), nil
}
func (p *GeminiProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	u := p.baseURL + "/models"
	var wire struct {
		Models []struct {
			Name            string `json:"name"`
			DisplayName     string `json:"displayName"`
			InputTokenLimit *int64 `json:"inputTokenLimit"`
		} `json:"models"`
	}
	if err := p.doGeminiJSON(ctx, http.MethodGet, u, nil, &wire); err != nil {
		return nil, err
	}
	out := make([]ModelInfo, 0, len(wire.Models))
	for _, m := range wire.Models {
		id := strings.TrimPrefix(m.Name, "models/")
		out = append(out, ModelInfo{ID: id, OwnedBy: "google", Object: "model", ContextLength: m.InputTokenLimit, DisplayName: m.DisplayName})
	}
	return out, nil
}

func (p *GeminiProvider) doGeminiJSON(ctx context.Context, method, u string, body any, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, rdr)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	p.addGeminiAuth(req)
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

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text         string              `json:"text"`
				FunctionCall *geminiFunctionCall `json:"functionCall"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason *string `json:"finishReason"`
		Index        *uint32 `json:"index"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     uint32 `json:"promptTokenCount"`
		CandidatesTokenCount uint32 `json:"candidatesTokenCount"`
		TotalTokenCount      uint32 `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func fromGeminiResponse(r geminiResponse, model string) ChatResponse {
	choices := make([]ChatChoice, 0, len(r.Candidates))
	for _, c := range r.Candidates {
		var text string
		var calls []ToolCall
		for _, p := range c.Content.Parts {
			text += p.Text
			if p.FunctionCall != nil {
				args, _ := json.Marshal(p.FunctionCall.Args)
				calls = append(calls, ToolCall{ID: p.FunctionCall.Name, Type: "function", Function: &FunctionCall{Name: p.FunctionCall.Name, Arguments: string(args)}})
			}
		}
		msg := AssistantMessage(text)
		msg.ToolCalls = calls
		idx := uint32(0)
		if c.Index != nil {
			idx = *c.Index
		}
		choices = append(choices, ChatChoice{Index: idx, Message: msg, FinishReason: c.FinishReason})
	}
	var usage *Usage
	if r.UsageMetadata != nil {
		usage = &Usage{PromptTokens: r.UsageMetadata.PromptTokenCount, CompletionTokens: r.UsageMetadata.CandidatesTokenCount, TotalTokens: r.UsageMetadata.TotalTokenCount}
	}
	return ChatResponse{Model: model, Choices: choices, Usage: usage}
}

func (p *GeminiProvider) StreamChat(ctx context.Context, req ChatRequest) (ChunkStream, error) {
	b, err := json.Marshal(toGeminiRequest(req))
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.geminiURL(req.Model, "streamGenerateContent", true), bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	p.addGeminiAuth(hreq)
	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return geminiStreamFromBody(resp.Body, req.Model), nil
}

func geminiStreamFromBody(body io.ReadCloser, model string) ChunkStream {
	return func(yield func(StreamChunk, error) bool) {
		defer body.Close()
		s := bufio.NewScanner(body)
		var data strings.Builder
		idx := 0
		emit := func(payload string) bool {
			var r geminiResponse
			if err := json.Unmarshal([]byte(payload), &r); err != nil {
				return yield(StreamChunk{}, err)
			}
			if len(r.Candidates) == 0 {
				return true
			}
			c := r.Candidates[0]
			var text string
			if len(c.Content.Parts) > 0 {
				text = c.Content.Parts[0].Text
			}
			idx++
			chunk := StreamChunk{ID: fmt.Sprintf("gemini-chunk-%d", idx), Model: model, Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: c.FinishReason}}}
			if text != "" {
				chunk.Choices[0].Delta.Content = &text
			}
			return yield(chunk, nil)
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
