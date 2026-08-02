package yongle

import "fmt"

type ChatRequest struct {
	Model               string             `json:"model"`
	Messages            []ChatMessage      `json:"messages"`
	Stream              bool               `json:"stream"`
	MaxTokens           *uint32            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *uint32            `json:"max_completion_tokens,omitempty"`
	Temperature         *float32           `json:"temperature,omitempty"`
	TopP                *float32           `json:"top_p,omitempty"`
	Modalities          []string           `json:"modalities,omitempty"`
	AudioOutput         *AudioOutputConfig `json:"audio,omitempty"`
	Tools               []Tool             `json:"tools,omitempty"`
	ToolChoice          *ToolChoice        `json:"tool_choice,omitempty"`
}

type AudioOutputConfig struct {
	Voice  string `json:"voice"`
	Format string `json:"format"`
}

type ChatRequestBuilder struct{ req ChatRequest }

func NewChatRequest(model string) *ChatRequestBuilder {
	return &ChatRequestBuilder{req: ChatRequest{Model: model}}
}
func (b *ChatRequestBuilder) Message(m ChatMessage) *ChatRequestBuilder {
	b.req.Messages = append(b.req.Messages, m)
	return b
}
func (b *ChatRequestBuilder) Messages(messages ...ChatMessage) *ChatRequestBuilder {
	b.req.Messages = append([]ChatMessage(nil), messages...)
	return b
}
func (b *ChatRequestBuilder) System(text string) *ChatRequestBuilder {
	return b.Message(SystemMessage(text))
}
func (b *ChatRequestBuilder) User(text string) *ChatRequestBuilder {
	return b.Message(UserMessage(text))
}
func (b *ChatRequestBuilder) Assistant(text string) *ChatRequestBuilder {
	return b.Message(AssistantMessage(text))
}
func (b *ChatRequestBuilder) Stream(stream bool) *ChatRequestBuilder { b.req.Stream = stream; return b }
func (b *ChatRequestBuilder) MaxTokens(n uint32) *ChatRequestBuilder { b.req.MaxTokens = &n; return b }

// MaxCompletionTokens sets OpenAI's max_completion_tokens (required for o-series /
// reasoning models where max_tokens is rejected). Other providers ignore unknown fields.
func (b *ChatRequestBuilder) MaxCompletionTokens(n uint32) *ChatRequestBuilder {
	b.req.MaxCompletionTokens = &n
	return b
}
func (b *ChatRequestBuilder) Temperature(v float32) *ChatRequestBuilder {
	b.req.Temperature = &v
	return b
}
func (b *ChatRequestBuilder) TopP(v float32) *ChatRequestBuilder { b.req.TopP = &v; return b }
func (b *ChatRequestBuilder) Modalities(modalities ...string) *ChatRequestBuilder {
	b.req.Modalities = append([]string(nil), modalities...)
	return b
}
func (b *ChatRequestBuilder) AudioOutput(voice, format string) *ChatRequestBuilder {
	b.req.AudioOutput = &AudioOutputConfig{Voice: voice, Format: format}
	return b
}
func (b *ChatRequestBuilder) Tool(tool Tool) *ChatRequestBuilder {
	b.req.Tools = append(b.req.Tools, tool)
	return b
}
func (b *ChatRequestBuilder) Tools(tools ...Tool) *ChatRequestBuilder {
	b.req.Tools = append([]Tool(nil), tools...)
	return b
}
func (b *ChatRequestBuilder) ToolChoice(choice ToolChoice) *ChatRequestBuilder {
	b.req.ToolChoice = &choice
	return b
}
func (b *ChatRequestBuilder) Build() (ChatRequest, error) {
	if b.req.Model == "" {
		return ChatRequest{}, fmt.Errorf("model must be specified")
	}
	if len(b.req.Messages) == 0 {
		return ChatRequest{}, fmt.Errorf("at least one message is required")
	}
	return b.req, nil
}

type TTSRequest struct {
	Model  string   `json:"model"`
	Input  string   `json:"input"`
	Voice  string   `json:"voice"`
	Format *string  `json:"response_format,omitempty"`
	Speed  *float32 `json:"speed,omitempty"`
}

type TTSRequestBuilder struct{ req TTSRequest }

func NewTTSRequest(model, input string) *TTSRequestBuilder {
	return &TTSRequestBuilder{req: TTSRequest{Model: model, Input: input}}
}
func (b *TTSRequestBuilder) Voice(voice string) *TTSRequestBuilder { b.req.Voice = voice; return b }
func (b *TTSRequestBuilder) Format(format string) *TTSRequestBuilder {
	b.req.Format = &format
	return b
}
func (b *TTSRequestBuilder) Speed(speed float32) *TTSRequestBuilder { b.req.Speed = &speed; return b }
func (b *TTSRequestBuilder) Build() (TTSRequest, error) {
	if b.req.Model == "" {
		return TTSRequest{}, fmt.Errorf("model required")
	}
	if b.req.Input == "" {
		return TTSRequest{}, fmt.Errorf("input required")
	}
	if b.req.Voice == "" {
		return TTSRequest{}, fmt.Errorf("voice required")
	}
	return b.req, nil
}
