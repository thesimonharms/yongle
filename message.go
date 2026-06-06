package yongle

import "encoding/json"

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type MessageContent struct {
	text  *string
	parts []ContentPart
}

func TextContent(s string) MessageContent              { return MessageContent{text: &s} }
func PartsContent(parts ...ContentPart) MessageContent { return MessageContent{parts: parts} }
func (c MessageContent) Text() string {
	if c.text != nil {
		return *c.text
	}
	out := ""
	for _, p := range c.parts {
		if p.Type == "text" {
			out += p.Text
		}
	}
	return out
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	if c.text != nil {
		return json.Marshal(c.Text())
	}
	return json.Marshal(c.parts)
}
func (c *MessageContent) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*c = TextContent("")
		return nil
	}
	var s string
	if json.Unmarshal(b, &s) == nil {
		*c = TextContent(s)
		return nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(b, &parts); err != nil {
		return err
	}
	*c = PartsContent(parts...)
	return nil
}

type ContentPart struct {
	Type       string         `json:"type"`
	Text       string         `json:"text,omitempty"`
	ImageURL   *ImageURL      `json:"image_url,omitempty"`
	InputAudio *AudioInput    `json:"input_audio,omitempty"`
	Document   *DocumentInput `json:"document,omitempty"`
}

func TextPart(text string) ContentPart { return ContentPart{Type: "text", Text: text} }
func ImagePart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url}}
}
func AudioPart(data, format string) ContentPart {
	return ContentPart{Type: "input_audio", InputAudio: &AudioInput{Data: data, Format: format}}
}
func DocumentPart(data, mediaType string) ContentPart {
	return ContentPart{Type: "document", Document: &DocumentInput{Data: data, MediaType: mediaType}}
}

type ImageURL struct {
	URL    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}

type AudioInput struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

func (a AudioInput) MIMEType() string {
	switch a.Format {
	case "mp3":
		return "audio/mpeg"
	case "flac":
		return "audio/flac"
	case "opus":
		return "audio/ogg; codecs=opus"
	case "aac":
		return "audio/aac"
	case "pcm16":
		return "audio/pcm"
	default:
		return "audio/wav"
	}
}

type DocumentInput struct {
	Data      string `json:"data"`
	MediaType string `json:"media_type"`
}
type AudioOutput struct {
	ID         *string `json:"id,omitempty"`
	Data       string  `json:"data"`
	ExpiresAt  *uint64 `json:"expires_at,omitempty"`
	Transcript *string `json:"transcript,omitempty"`
}

type ChatMessage struct {
	Role       Role           `json:"role"`
	Content    MessageContent `json:"content"`
	Audio      *AudioOutput   `json:"audio,omitempty"`
	ToolCalls  []ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

func SystemMessage(content string) ChatMessage {
	return ChatMessage{Role: RoleSystem, Content: TextContent(content)}
}
func UserMessage(content string) ChatMessage {
	return ChatMessage{Role: RoleUser, Content: TextContent(content)}
}
func AssistantMessage(content string) ChatMessage {
	return ChatMessage{Role: RoleAssistant, Content: TextContent(content)}
}
func ToolResultMessage(id, content string) ChatMessage {
	return ChatMessage{Role: RoleTool, Content: TextContent(content), ToolCallID: id}
}
func UserMessageWithImage(text, imageURL string) ChatMessage {
	return ChatMessage{Role: RoleUser, Content: PartsContent(TextPart(text), ImagePart(imageURL))}
}
func MessageWithParts(role Role, parts ...ContentPart) ChatMessage {
	return ChatMessage{Role: role, Content: PartsContent(parts...)}
}
