package yongle

import (
	"encoding/base64"
	"strings"
)

func parseDataURL(url string) (mimeType, data string, ok bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	comma := strings.IndexByte(url, ',')
	if comma < 0 {
		return "", "", false
	}
	meta := strings.TrimPrefix(url[:comma], "data:")
	payload := url[comma+1:]
	parts := strings.Split(meta, ";")
	mimeType = parts[0]
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	return mimeType, payload, true
}

func guessImageMIMEType(url string) string {
	lower := strings.ToLower(url)
	switch {
	case strings.Contains(lower, ".png"):
		return "image/png"
	case strings.Contains(lower, ".webp"):
		return "image/webp"
	case strings.Contains(lower, ".gif"):
		return "image/gif"
	default:
		return "image/jpeg"
	}
}

func jsonObjectFromString(s string) map[string]any {
	// Provider APIs generally require an object for tool results. Keep plain text
	// round-trippable under result.
	return map[string]any{"result": s}
}

func mustBase64Decode(s string) []byte {
	b, _ := base64.StdEncoding.DecodeString(s)
	return b
}
