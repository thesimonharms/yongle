package yongle

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

type OpenAIProvider struct{ *OpenAICompatibleProvider }

func (p *OpenAIProvider) TTS(ctx context.Context, req TTSRequest) ([]byte, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/audio/speech", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
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
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &Error{Provider: p.providerName, StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return io.ReadAll(resp.Body)
}
