package yongle

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Provider is the common interface implemented by every AI backend.
type Provider interface {
	Name() string
	Chat(context.Context, ChatRequest) (ChatResponse, error)
	StreamChat(context.Context, ChatRequest) (ChunkStream, error)
	Models(context.Context) ([]ModelInfo, error)
	TTS(context.Context, TTSRequest) ([]byte, error)
}

// ChunkStream is a pull-friendly Go iterator over streaming chunks.
type ChunkStream func(yield func(StreamChunk, error) bool)

type Error struct {
	Provider   string
	StatusCode int
	Message    string
}

func (e *Error) Error() string {
	if e.Provider != "" && e.StatusCode != 0 {
		return fmt.Sprintf("%s API error (%d): %s", e.Provider, e.StatusCode, e.Message)
	}
	if e.Provider != "" {
		return fmt.Sprintf("%s API error: %s", e.Provider, e.Message)
	}
	return e.Message
}

var ErrUnsupported = errors.New("unsupported operation")

// ClientOption customizes providers created by constructors.
type ClientOption func(*clientConfig)

type clientConfig struct {
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

func defaultHTTPClient() *http.Client { return &http.Client{Timeout: 60 * time.Second} }

func WithBaseURL(url string) ClientOption {
	return func(c *clientConfig) { c.baseURL = strings.TrimRight(url, "/") }
}
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *clientConfig) {
		if client != nil {
			c.httpClient = client
		}
	}
}
func WithHeader(k, v string) ClientOption {
	return func(c *clientConfig) {
		if c.headers == nil {
			c.headers = map[string]string{}
		}
		c.headers[k] = v
	}
}

func applyOptions(defaultBase string, opts ...ClientOption) clientConfig {
	cfg := clientConfig{baseURL: strings.TrimRight(defaultBase, "/"), httpClient: defaultHTTPClient(), headers: map[string]string{}}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func doJSON(ctx context.Context, client *http.Client, method, url, apiKey string, headers map[string]string, body any, out any) error {
	return doJSONProvider(ctx, client, "", method, url, apiKey, headers, body, out)
}

func doJSONProvider(ctx context.Context, client *http.Client, provider, method, url, apiKey string, headers map[string]string, body any, out any) error {
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
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return &Error{Provider: provider, StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(b))}
	}
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}
