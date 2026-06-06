package yongle

type ChatResponse struct {
	ID        string       `json:"id"`
	Model     string       `json:"model"`
	Choices   []ChatChoice `json:"choices"`
	Usage     *Usage       `json:"usage,omitempty"`
	Citations []string     `json:"citations,omitempty"`
}

func (r ChatResponse) Content() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.Content.Text()
}

type ChatChoice struct {
	Index        uint32      `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason *string     `json:"finish_reason"`
}
type Usage struct {
	PromptTokens     uint32 `json:"prompt_tokens"`
	CompletionTokens uint32 `json:"completion_tokens"`
	TotalTokens      uint32 `json:"total_tokens"`
}

type StreamChunk struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

func (c StreamChunk) DeltaContent() string {
	if len(c.Choices) == 0 || c.Choices[0].Delta.Content == nil {
		return ""
	}
	return *c.Choices[0].Delta.Content
}
func (c StreamChunk) IsFinished() bool { return len(c.Choices) > 0 && c.Choices[0].FinishReason != nil }

type StreamChoice struct {
	Index        uint32  `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}
type Delta struct {
	Role      *Role           `json:"role,omitempty"`
	Content   *string         `json:"content,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
	Created *int64 `json:"created,omitempty"`
}

type modelListResponse struct {
	Data []ModelInfo `json:"data"`
}
