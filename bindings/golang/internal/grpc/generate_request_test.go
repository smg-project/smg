package grpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/lightseek/smg/go-grpc-sdk/internal/ffi"
	"github.com/lightseek/smg/go-grpc-sdk/internal/proto"
)

// A scheduler client that keeps the generate requests it is handed and
// answers each with a stream that has already ended.
type capturingScheduler struct {
	proto.SglangSchedulerClient
	mu       sync.Mutex
	requests []*proto.GenerateRequest
}

func (c *capturingScheduler) Generate(_ context.Context, in *proto.GenerateRequest, _ ...grpc.CallOption) (grpc.ServerStreamingClient[proto.GenerateResponse], error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, in)
	return endedGenerateStream{}, nil
}

func (c *capturingScheduler) last() *proto.GenerateRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.requests) == 0 {
		return nil
	}
	return c.requests[len(c.requests)-1]
}

// A generate stream that ends before its first response.
type endedGenerateStream struct{ grpc.ClientStream }

func (endedGenerateStream) Recv() (*proto.GenerateResponse, error) { return nil, io.EOF }
func (endedGenerateStream) CloseSend() error                       { return nil }

// A tokenizer directory for the tests: a word-level vocabulary and a chat
// template that writes one `role: content` line per message, so the
// preprocessor renders and tokenizes a request without a checkpoint.
func testTokenizerDir(t *testing.T) string {
	t.Helper()
	vocab := map[string]int{"[UNK]": 0}
	for b := 0; b < 256; b++ {
		vocab[fmt.Sprintf("<0x%02X>", b)] = b + 1
	}
	definition, err := json.Marshal(map[string]interface{}{
		"version":      "1.0",
		"added_tokens": []interface{}{},
		"model":        map[string]interface{}{"type": "WordLevel", "unk_token": "[UNK]", "vocab": vocab},
		"decoder": map[string]interface{}{"type": "Sequence", "decoders": []interface{}{
			map[string]interface{}{"type": "ByteFallback"},
			map[string]interface{}{"type": "Fuse"},
		}},
	})
	if err != nil {
		t.Fatalf("tokenizer definition: %v", err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), definition, 0o644); err != nil {
		t.Fatalf("tokenizer.json: %v", err)
	}
	template := "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}" +
		"{% if add_generation_prompt %}assistant: {% endif %}"
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(template), 0o644); err != nil {
		t.Fatalf("chat_template.jinja: %v", err)
	}
	return dir
}

// The engine's skip_special_tokens follows the request's tools: kept (false)
// when a response parser reads the engine's tool-call markers, that is tools
// under a choice that allows calls and no JSON-schema constraint; otherwise
// the request's own setting, which skips them unless the request says not to.
func TestGenerateRequestSkipsSpecialTokensAsTheParserNeeds(t *testing.T) {
	handle, err := ffi.CreateTokenizerHandle(testTokenizerDir(t))
	if err != nil {
		t.Fatalf("tokenizer handle: %v", err)
	}
	defer ffi.FreeTokenizerHandle(handle)
	scheduler := &capturingScheduler{}
	client := &GrpcClient{
		client:          scheduler,
		tokenizerHandle: handle,
		bufferSizes:     ChannelBufferSizes{ResultJSONChan: 16, ErrChan: 4, RecvChan: 4},
		timeouts:        Timeouts{CloseTimeout: 5 * time.Second},
	}

	const messages = `"messages":[{"role":"user","content":"hi"}]`
	const tools = `"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]`
	for _, tc := range []struct {
		name     string
		request  string
		wantSkip bool
	}{
		{name: "tools, no choice", request: `{"model":"m",` + messages + `,` + tools + `}`, wantSkip: false},
		{name: "tools under auto", request: `{"model":"m",` + messages + `,` + tools + `,"tool_choice":"auto"}`, wantSkip: false},
		{name: "tools under none", request: `{"model":"m",` + messages + `,` + tools + `,"tool_choice":"none"}`, wantSkip: true},
		{name: "a named function, held to its schema", request: `{"model":"m",` + messages + `,` + tools + `,"tool_choice":{"type":"function","function":{"name":"get_weather"}}}`, wantSkip: true},
		{name: "no tools", request: `{"model":"m",` + messages + `}`, wantSkip: true},
		{name: "an empty tools list", request: `{"model":"m",` + messages + `,"tools":[]}`, wantSkip: true},
		{name: "no tools, the request keeps them", request: `{"model":"m",` + messages + `,"skip_special_tokens":false}`, wantSkip: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := client.CreateChatCompletionStream(context.Background(), tc.request)
			if err != nil {
				t.Fatalf("CreateChatCompletionStream: %v", err)
			}
			defer stream.Close()
			got := scheduler.last()
			if got == nil || got.SamplingParams == nil {
				t.Fatal("no generate request reached the scheduler")
			}
			if got.SamplingParams.SkipSpecialTokens != tc.wantSkip {
				t.Fatalf("skip_special_tokens = %v, want %v", got.SamplingParams.SkipSpecialTokens, tc.wantSkip)
			}
		})
	}
}
