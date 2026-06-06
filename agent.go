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

const defaultHermesAgentBaseURL = "http://127.0.0.1:8642/v1"

// AgentProvider is the common interface for stateful agent runtimes.
//
// Agent providers are intentionally separate from chat Providers: they can run
// multi-step tasks, preserve server-side response state, stream lifecycle/tool
// events, and return OpenAI Responses-compatible response objects.
type AgentProvider interface {
	Name() string
	Run(context.Context, AgentRunRequest) (AgentResponse, error)
	StreamRun(context.Context, AgentRunRequest) (AgentEventStream, error)
}

// AgentEventStream is a pull-friendly Go iterator over agent SSE events.
type AgentEventStream func(yield func(AgentEvent, error) bool)

// AgentInput is the input accepted by an OpenAI Responses-compatible agent
// endpoint. It serializes as either a plain text string or an array of
// structured input items.
type AgentInput struct {
	Text  string
	Items []AgentInputItem
}

func AgentTextInput(text string) AgentInput { return AgentInput{Text: text} }
func AgentItemInput(items ...AgentInputItem) AgentInput {
	return AgentInput{Items: append([]AgentInputItem(nil), items...)}
}

func (i AgentInput) isEmpty() bool {
	if len(i.Items) > 0 {
		return false
	}
	return strings.TrimSpace(i.Text) == ""
}

func (i AgentInput) MarshalJSON() ([]byte, error) {
	if len(i.Items) > 0 {
		return json.Marshal(i.Items)
	}
	return json.Marshal(i.Text)
}

func (i *AgentInput) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		i.Text = text
		i.Items = nil
		return nil
	}
	var items []AgentInputItem
	if err := json.Unmarshal(data, &items); err != nil {
		return err
	}
	i.Text = ""
	i.Items = items
	return nil
}

type AgentInputItem struct {
	Role    string            `json:"role"`
	Content AgentInputContent `json:"content"`
}

type AgentInputContent struct {
	Text  string
	Parts []AgentInputContentPart
}

func AgentTextContent(text string) AgentInputContent { return AgentInputContent{Text: text} }
func AgentPartContent(parts ...AgentInputContentPart) AgentInputContent {
	return AgentInputContent{Parts: append([]AgentInputContentPart(nil), parts...)}
}

func (c AgentInputContent) MarshalJSON() ([]byte, error) {
	if len(c.Parts) > 0 {
		return json.Marshal(c.Parts)
	}
	return json.Marshal(c.Text)
}

func (c *AgentInputContent) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		c.Text = text
		c.Parts = nil
		return nil
	}
	var parts []AgentInputContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return err
	}
	c.Text = ""
	c.Parts = parts
	return nil
}

type AgentInputContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// AgentRunRequest is a request to a stateful agent runtime.
type AgentRunRequest struct {
	Model              string     `json:"model"`
	Input              AgentInput `json:"input"`
	Instructions       *string    `json:"instructions,omitempty"`
	PreviousResponseID *string    `json:"previous_response_id,omitempty"`
	Conversation       *string    `json:"conversation,omitempty"`
	Store              *bool      `json:"store,omitempty"`
	Stream             bool       `json:"stream"`
}

type AgentRunRequestBuilder struct{ req AgentRunRequest }

func NewAgentRunRequest(model, input string) *AgentRunRequestBuilder {
	return &AgentRunRequestBuilder{req: AgentRunRequest{Model: model, Input: AgentTextInput(input)}}
}
func NewAgentRunRequestWithInput(model string, input AgentInput) *AgentRunRequestBuilder {
	return &AgentRunRequestBuilder{req: AgentRunRequest{Model: model, Input: input}}
}
func (b *AgentRunRequestBuilder) Instructions(instructions string) *AgentRunRequestBuilder {
	b.req.Instructions = &instructions
	return b
}
func (b *AgentRunRequestBuilder) PreviousResponseID(id string) *AgentRunRequestBuilder {
	b.req.PreviousResponseID = &id
	return b
}
func (b *AgentRunRequestBuilder) Conversation(conversation string) *AgentRunRequestBuilder {
	b.req.Conversation = &conversation
	return b
}
func (b *AgentRunRequestBuilder) Store(store bool) *AgentRunRequestBuilder {
	b.req.Store = &store
	return b
}
func (b *AgentRunRequestBuilder) Stream(stream bool) *AgentRunRequestBuilder {
	b.req.Stream = stream
	return b
}
func (b *AgentRunRequestBuilder) Build() (AgentRunRequest, error) {
	if b.req.Model == "" {
		return AgentRunRequest{}, fmt.Errorf("model must be specified")
	}
	if b.req.Input.isEmpty() {
		return AgentRunRequest{}, fmt.Errorf("input must not be empty")
	}
	return b.req, nil
}

// AgentResponse is a completed response from an agent runtime.
type AgentResponse struct {
	ID     string            `json:"id"`
	Object *string           `json:"object,omitempty"`
	Status string            `json:"status"`
	Model  *string           `json:"model,omitempty"`
	Output []AgentOutputItem `json:"output,omitempty"`
	Usage  *AgentUsage       `json:"usage,omitempty"`
}

func (r AgentResponse) OutputText() string {
	var b strings.Builder
	for _, item := range r.Output {
		for _, content := range item.Content {
			if content.Type == "output_text" {
				b.WriteString(content.Text)
			}
		}
	}
	return b.String()
}

type AgentOutputItem struct {
	Type      string               `json:"type"`
	Role      *string              `json:"role,omitempty"`
	Name      *string              `json:"name,omitempty"`
	Arguments *string              `json:"arguments,omitempty"`
	CallID    *string              `json:"call_id,omitempty"`
	Output    *string              `json:"output,omitempty"`
	Content   []AgentOutputContent `json:"content,omitempty"`
}

type AgentOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type AgentUsage struct {
	InputTokens  uint32 `json:"input_tokens"`
	OutputTokens uint32 `json:"output_tokens"`
	TotalTokens  uint32 `json:"total_tokens"`
}

// AgentEvent is one server-sent event from an agent run.
type AgentEvent struct {
	Event string
	Data  map[string]any
}

func (e AgentEvent) OutputTextDelta() string {
	if delta, ok := e.Data["delta"].(string); ok {
		return delta
	}
	return ""
}

type HermesAgentProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	headers    map[string]string
}

func NewHermesAgentProvider(apiKey string, opts ...ClientOption) *HermesAgentProvider {
	cfg := applyOptions(defaultHermesAgentBaseURL, opts...)
	return &HermesAgentProvider{baseURL: cfg.baseURL, apiKey: apiKey, httpClient: cfg.httpClient, headers: cfg.headers}
}

func NewHermesAgentProviderNoKey(opts ...ClientOption) *HermesAgentProvider {
	return NewHermesAgentProvider("", opts...)
}

func (p *HermesAgentProvider) Name() string { return "hermes-agent" }

func (p *HermesAgentProvider) Run(ctx context.Context, req AgentRunRequest) (AgentResponse, error) {
	req.Stream = false
	var out AgentResponse
	err := doJSON(ctx, p.httpClient, http.MethodPost, p.responsesURL(), p.apiKey, p.headers, req, &out)
	return out, err
}

func (p *HermesAgentProvider) StreamRun(ctx context.Context, req AgentRunRequest) (AgentEventStream, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.responsesURL(), bytes.NewReader(body))
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
		return nil, &Error{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return agentEventsFromBody(resp.Body), nil
}

func (p *HermesAgentProvider) responsesURL() string {
	return strings.TrimRight(p.baseURL, "/") + "/responses"
}

func agentEventsFromBody(body io.ReadCloser) AgentEventStream {
	return func(yield func(AgentEvent, error) bool) {
		defer body.Close()
		events, errs := parseAgentSSE(bufio.NewReader(body))
		for event := range events {
			if !yield(event, nil) {
				return
			}
		}
		if err := <-errs; err != nil {
			yield(AgentEvent{}, err)
		}
	}
}

func parseAgentSSE(r *bufio.Reader) (<-chan AgentEvent, <-chan error) {
	events := make(chan AgentEvent)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		var data strings.Builder
		var currentEvent string
		flush := func() bool {
			if data.Len() == 0 {
				currentEvent = ""
				return true
			}
			payload := strings.TrimSpace(data.String())
			data.Reset()
			if payload == "[DONE]" {
				errs <- nil
				return false
			}
			var value map[string]any
			if err := json.Unmarshal([]byte(payload), &value); err != nil {
				errs <- err
				return false
			}
			events <- AgentEvent{Event: currentEvent, Data: value}
			currentEvent = ""
			return true
		}
		for {
			line, err := r.ReadString('\n')
			if err != nil && err != io.EOF {
				errs <- err
				return
			}
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "":
				if !flush() {
					return
				}
			case strings.HasPrefix(line, "event:"):
				currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			case strings.HasPrefix(line, "data:"):
				if data.Len() > 0 {
					data.WriteByte('\n')
				}
				data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
			if err == io.EOF {
				if data.Len() > 0 && !flush() {
					return
				}
				errs <- nil
				return
			}
		}
	}()
	return events, errs
}
