package yongle

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

func parseSSE(r *bufio.Reader) (<-chan StreamChunk, <-chan error) {
	chunks := make(chan StreamChunk)
	errs := make(chan error, 1)
	go func() {
		defer close(chunks)
		defer close(errs)
		var data strings.Builder
		for {
			line, err := r.ReadString('\n')
			if err != nil && err != io.EOF {
				errs <- err
				return
			}
			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				if data.Len() > 0 {
					payload := strings.TrimSpace(data.String())
					data.Reset()
					if payload == "[DONE]" {
						errs <- nil
						return
					}
					var chunk StreamChunk
					if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
						errs <- err
						return
					}
					chunks <- chunk
				}
			} else if strings.HasPrefix(line, "data:") {
				if data.Len() > 0 {
					data.WriteByte('\n')
				}
				data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
			if err == io.EOF {
				if data.Len() > 0 {
					payload := strings.TrimSpace(data.String())
					if payload != "[DONE]" {
						var chunk StreamChunk
						if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
							errs <- err
							return
						}
						chunks <- chunk
					}
				}
				errs <- nil
				return
			}
		}
	}()
	return chunks, errs
}

func streamFromBody(body io.ReadCloser) ChunkStream {
	return func(yield func(StreamChunk, error) bool) {
		defer body.Close()
		chunks, errs := parseSSE(bufio.NewReader(body))
		for chunk := range chunks {
			if !yield(chunk, nil) {
				return
			}
		}
		if err := <-errs; err != nil {
			yield(StreamChunk{}, err)
		}
	}
}
