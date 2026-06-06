package yongle

import "encoding/json"

type FunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type Tool struct {
	Type     string              `json:"type"`
	Function *FunctionDefinition `json:"function,omitempty"`
}

func FunctionTool(name, description string, parameters map[string]any) Tool {
	return Tool{Type: "function", Function: &FunctionDefinition{Name: name, Description: description, Parameters: parameters}}
}

type ToolChoice struct{ value any }

func ToolChoiceAuto() ToolChoice     { return ToolChoice{value: "auto"} }
func ToolChoiceNone() ToolChoice     { return ToolChoice{value: "none"} }
func ToolChoiceRequired() ToolChoice { return ToolChoice{value: "required"} }
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{value: map[string]any{"type": "function", "function": map[string]string{"name": name}}}
}
func (t ToolChoice) MarshalJSON() ([]byte, error) { return json.Marshal(t.value) }

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolCall struct {
	ID       string        `json:"id"`
	Type     string        `json:"type"`
	Function *FunctionCall `json:"function,omitempty"`
}

type ToolCallDelta struct {
	Index    int           `json:"index"`
	ID       string        `json:"id,omitempty"`
	Type     string        `json:"type,omitempty"`
	Function *FunctionCall `json:"function,omitempty"`
}
