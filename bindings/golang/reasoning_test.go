package smg

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

// A stream of recorded chunks, as the converter sends them.
type recordedChunks struct {
	chunks []string
}

func (s *recordedChunks) RecvJSON() (string, error) {
	if len(s.chunks) == 0 {
		return "", io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

// A thinking model's thought arrives as reasoning_content, apart from the
// content; the non-streaming response keeps both, with the last chunk's
// finish reason and usage.
func TestNonStreamingResponseKeepsTheReasoningApartFromTheContent(t *testing.T) {
	stream := &recordedChunks{chunks: []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Paris "}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"it is."}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"content":"Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
	}}
	resp, err := collectChatCompletion(stream)
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	message := resp.Choices[0].Message
	if message.ReasoningContent != "Paris it is." || message.Content != "Paris." {
		t.Fatalf("message %+v", message)
	}
	if resp.Choices[0].FinishReason != "stop" || resp.Usage.TotalTokens != 7 || resp.ID != "chatcmpl-1" {
		t.Fatalf("response %+v", resp)
	}
}

// A response without a thought carries no reasoning field: a client that
// never asked for one sees the shape it saw before.
func TestNonStreamingResponseWithoutAThoughtOmitsTheReasoning(t *testing.T) {
	stream := &recordedChunks{chunks: []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}`,
	}}
	resp, err := collectChatCompletion(stream)
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	message := resp.Choices[0].Message
	if message.Content != "Hello" || message.ReasoningContent != "" {
		t.Fatalf("message %+v", message)
	}
	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if strings.Contains(string(encoded), "reasoning_content") {
		t.Fatalf("an empty thought is encoded: %s", encoded)
	}
}
