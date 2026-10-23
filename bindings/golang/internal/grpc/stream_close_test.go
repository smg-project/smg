package grpc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/lightseek/smg/go-grpc-sdk/internal/ffi"
	"github.com/lightseek/smg/go-grpc-sdk/internal/proto"
)

// A scheduler client whose generate streams play a script: the responses in
// order, then the terminal error (io.EOF for a stream that ends on its own).
// Without a terminal error the stream stays open until the caller closes it.
type scriptedScheduler struct {
	proto.SglangSchedulerClient
	responses []*proto.GenerateResponse
	terminal  error
}

func (s *scriptedScheduler) Generate(ctx context.Context, _ *proto.GenerateRequest, _ ...grpc.CallOption) (grpc.ServerStreamingClient[proto.GenerateResponse], error) {
	return &scriptedGenerateStream{
		ctx:       ctx,
		responses: s.responses,
		terminal:  s.terminal,
		closed:    make(chan struct{}),
	}, nil
}

// GetModelInfo answers the client's one-time weight-version lookup at the
// first stream; the embedded nil client would panic on it.
func (s *scriptedScheduler) GetModelInfo(_ context.Context, _ *proto.GetModelInfoRequest, _ ...grpc.CallOption) (*proto.GetModelInfoResponse, error) {
	return &proto.GetModelInfoResponse{WeightVersion: "v-test"}, nil
}

type scriptedGenerateStream struct {
	grpc.ClientStream
	ctx       context.Context
	responses []*proto.GenerateResponse
	terminal  error
	closed    chan struct{}
	closeOnce sync.Once
	mu        sync.Mutex
	next      int
}

func (s *scriptedGenerateStream) Recv() (*proto.GenerateResponse, error) {
	s.mu.Lock()
	if s.next < len(s.responses) {
		resp := s.responses[s.next]
		s.next++
		s.mu.Unlock()
		return resp, nil
	}
	s.mu.Unlock()
	if s.terminal != nil {
		return nil, s.terminal
	}
	select {
	case <-s.closed:
		return nil, io.EOF
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

func (s *scriptedGenerateStream) CloseSend() error {
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}

// A tokenizer directory for the tests: a byte-level word vocabulary and a chat
// template that writes one `role: content` line per message, so the
// preprocessor renders and tokenizes a request without a checkpoint.
func closeTestTokenizerDir(t *testing.T) string {
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

// The engine's answer to a request: one chunk of two byte tokens ("Hi" in
// the test vocabulary) and the completion that ends the stream.
func scriptedAnswer() []*proto.GenerateResponse {
	return []*proto.GenerateResponse{
		{Response: &proto.GenerateResponse_Chunk{Chunk: &proto.GenerateStreamChunk{
			TokenIds: []uint32{0x48 + 1, 0x69 + 1}, PromptTokens: 4, CompletionTokens: 2,
		}}},
		{Response: &proto.GenerateResponse_Complete{Complete: &proto.GenerateComplete{
			OutputIds: []uint32{0x48 + 1, 0x69 + 1}, FinishReason: "stop", PromptTokens: 4, CompletionTokens: 2,
		}}},
	}
}

const closeTestRequest = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`

func newCloseTestClient(t *testing.T, scheduler *scriptedScheduler, resultBuffer int) *GrpcClient {
	t.Helper()
	handle, err := ffi.CreateTokenizerHandle(closeTestTokenizerDir(t))
	if err != nil {
		t.Fatalf("tokenizer handle: %v", err)
	}
	t.Cleanup(func() { ffi.FreeTokenizerHandle(handle) })
	return &GrpcClient{
		client:          scheduler,
		tokenizerHandle: handle,
		bufferSizes:     ChannelBufferSizes{ResultJSONChan: resultBuffer, ErrChan: 4, RecvChan: 4},
		timeouts:        Timeouts{CloseTimeout: 5 * time.Second},
	}
}

// Reads the stream to its end and returns the error that ended it.
func drainStream(stream *GrpcChatCompletionStream) error {
	for {
		if _, err := stream.RecvJSON(); err != nil {
			return err
		}
	}
}

func waitForReadLoop(t *testing.T, stream *GrpcChatCompletionStream) {
	t.Helper()
	select {
	case <-stream.readLoopDone:
	case <-time.After(5 * time.Second):
		t.Fatal("the read loop did not finish")
	}
}

// Waits until the read loop has entered its cleanup (the point from which
// Close finds the flag set), without requiring the cleanup to finish.
func waitForReadLoopCleanup(t *testing.T, stream *GrpcChatCompletionStream) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for atomic.LoadInt32(&stream.closed) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the read loop did not reach its cleanup")
		}
		time.Sleep(time.Millisecond)
	}
}

// Close releases the converter handle exactly once, however the stream ended:
// on its own (the way every non-streaming completion and every stream read to
// EOF ends), with the engine's error, or stopped by Close while still open; a
// second Close, in sequence or concurrently, frees nothing more.
func TestCloseReleasesTheConverterOnEveryPath(t *testing.T) {
	engineErr := errors.New("engine gone")
	for _, tc := range []struct {
		name      string
		scheduler *scriptedScheduler
		// No room for results: a chunk processor waits for a reader.
		unbufferedResults bool
		drive             func(t *testing.T, stream *GrpcChatCompletionStream)
	}{
		{
			name:      "ended on its own, read to EOF, then Close",
			scheduler: &scriptedScheduler{responses: scriptedAnswer(), terminal: io.EOF},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				if err := drainStream(stream); err != io.EOF {
					t.Fatalf("stream ended with %v, want EOF", err)
				}
				_ = stream.Close()
			},
		},
		{
			name:      "ended on its own, Close without a read",
			scheduler: &scriptedScheduler{responses: scriptedAnswer(), terminal: io.EOF},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				waitForReadLoop(t, stream)
				_ = stream.Close()
			},
		},
		{
			name:      "ended with the engine's error",
			scheduler: &scriptedScheduler{responses: scriptedAnswer()[:1], terminal: engineErr},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				if err := drainStream(stream); !errors.Is(err, engineErr) {
					t.Fatalf("stream ended with %v, want the engine's error", err)
				}
				// The error reaches the reader before the loop's cleanup; let
				// the cleanup finish so Close meets a loop that is done.
				waitForReadLoop(t, stream)
				_ = stream.Close()
			},
		},
		{
			// The reader went away with chunks still to deliver: the loop's
			// cleanup is waiting on their processors when Close arrives.
			name:              "ended on its own, chunks waiting for a reader",
			scheduler:         &scriptedScheduler{responses: scriptedAnswer(), terminal: io.EOF},
			unbufferedResults: true,
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				waitForReadLoopCleanup(t, stream)
				_ = stream.Close()
			},
		},
		{
			name:      "closed while open",
			scheduler: &scriptedScheduler{responses: scriptedAnswer()[:1]},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				if _, err := stream.RecvJSON(); err != nil {
					t.Fatalf("first chunk: %v", err)
				}
				_ = stream.Close()
			},
		},
		{
			name:      "closed twice",
			scheduler: &scriptedScheduler{responses: scriptedAnswer(), terminal: io.EOF},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				_ = drainStream(stream)
				_ = stream.Close()
				_ = stream.Close()
			},
		},
		{
			name:      "closed twice, concurrently, while open",
			scheduler: &scriptedScheduler{},
			drive: func(t *testing.T, stream *GrpcChatCompletionStream) {
				var wg sync.WaitGroup
				for i := 0; i < 2; i++ {
					wg.Add(1)
					go func() {
						defer wg.Done()
						_ = stream.Close()
					}()
				}
				wg.Wait()
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resultBuffer := 16
			if tc.unbufferedResults {
				resultBuffer = 0
			}
			client := newCloseTestClient(t, tc.scheduler, resultBuffer)
			before := ffi.LiveGrpcResponseConverters()
			stream, err := client.CreateChatCompletionStream(context.Background(), closeTestRequest)
			if err != nil {
				t.Fatalf("CreateChatCompletionStream: %v", err)
			}
			if got := ffi.LiveGrpcResponseConverters() - before; got != 1 {
				t.Fatalf("live converters after open = %d, want 1", got)
			}
			tc.drive(t, stream)
			select {
			case <-stream.readLoopDone:
			default:
				t.Fatal("Close returned before the read loop finished")
			}
			if got := ffi.LiveGrpcResponseConverters() - before; got != 0 {
				t.Fatalf("live converters after Close = %d, want 0 (the handle leaked)", got)
			}
		})
	}
}

// The converter is freed under the lock every chunk takes: a chunk processor
// still inside it when Close frees (Close gave up waiting for the loop) finishes
// first, and a chunk that arrives after Close meets a closed postprocessor
// instead of freed memory.
func TestChunksAroundCloseMeetTheConverterOrAClosedPostprocessor(t *testing.T) {
	client := newCloseTestClient(t, &scriptedScheduler{}, 16)
	before := ffi.LiveGrpcResponseConverters()
	stream, err := client.CreateChatCompletionStream(context.Background(), closeTestRequest)
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	chunkJSON, err := protoToJSON(scriptedAnswer()[0])
	if err != nil {
		t.Fatalf("chunk JSON: %v", err)
	}
	// A processor that keeps feeding chunks while Close runs: every call takes
	// the converter under the postprocessor's lock, so it either completes on
	// a live converter or is refused after the free, never in between.
	stop := make(chan struct{})
	processed := make(chan int, 1)
	go func() {
		n := 0
		for {
			select {
			case <-stop:
				processed <- n
				return
			default:
			}
			if _, _, err := stream.batchPostprocessor.AddChunk(chunkJSON); err != nil {
				if !errors.Is(err, ffi.ErrPostprocessorClosed) {
					t.Errorf("a chunk in flight around Close failed with %v", err)
				}
				processed <- n
				return
			}
			n++
		}
	}()
	time.Sleep(10 * time.Millisecond)
	_ = stream.Close()
	close(stop)
	n := <-processed
	if n == 0 {
		t.Fatal("no chunk was processed before Close")
	}
	if got := ffi.LiveGrpcResponseConverters() - before; got != 0 {
		t.Fatalf("live converters after Close = %d, want 0", got)
	}
	if _, _, err := stream.batchPostprocessor.AddChunk(chunkJSON); !errors.Is(err, ffi.ErrPostprocessorClosed) {
		t.Fatalf("a chunk after Close got %v, want the closed-postprocessor error", err)
	}
	stream.batchPostprocessor.Close()
	if got := ffi.LiveGrpcResponseConverters() - before; got != 0 {
		t.Fatalf("live converters after a second Close = %d, want 0", got)
	}
}

// N requests completed the ordinary way (read to EOF, then Close) leave no
// converter handle behind.
func TestCompletedStreamsLeaveNoLiveConverters(t *testing.T) {
	const n = 50
	client := newCloseTestClient(t, &scriptedScheduler{responses: scriptedAnswer(), terminal: io.EOF}, 16)
	before := ffi.LiveGrpcResponseConverters()
	for i := 0; i < n; i++ {
		stream, err := client.CreateChatCompletionStream(context.Background(), closeTestRequest)
		if err != nil {
			t.Fatalf("request %d: CreateChatCompletionStream: %v", i, err)
		}
		if err := drainStream(stream); err != io.EOF {
			t.Fatalf("request %d: stream ended with %v, want EOF", i, err)
		}
		_ = stream.Close()
	}
	if leaked := ffi.LiveGrpcResponseConverters() - before; leaked != 0 {
		t.Fatalf("%d of %d completed streams left their converter handle live", leaked, n)
	}
}
