package grpc

import (
	"context"
	"errors"
	"io"
	"slices"
	"sync"
	"testing"
	"time"
)

func TestRecvJSONDrainsCompletedStream(t *testing.T) {
	chunks := []string{
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"content":" world"}}]}`,
		`{"choices":[{"finish_reason":"stop"}],"usage":{"completion_tokens":2}}`,
	}

	for _, cancelContext := range []bool{false, true} {
		name := "before_context_cancellation"
		if cancelContext {
			name = "after_context_cancellation"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			stream := newRecvJSONTestStream(ctx, len(chunks)+2)
			stream.resultJSONChan <- ""
			for _, chunk := range chunks {
				stream.resultJSONChan <- chunk
			}
			stream.resultJSONChan <- ""

			// Match readLoop's cleanup after the worker completes, before the
			// caller has consumed the buffered response chunks.
			close(stream.resultJSONChan)
			close(stream.errChan)
			if cancelContext {
				cancel()
			}

			for i, want := range chunks {
				got, err := stream.RecvJSON()
				if err != nil || got != want {
					t.Fatalf("chunk %d: RecvJSON() = (%q, %v), want (%q, nil)", i, got, err, want)
				}
			}
			if got, err := stream.RecvJSON(); got != "" || err != io.EOF {
				t.Fatalf("after final chunk: RecvJSON() = (%q, %v), want (\"\", EOF)", got, err)
			}
		})
	}
}

func TestRecvJSONSkipsEmptyChunks(t *testing.T) {
	stream := newRecvJSONTestStream(context.Background(), 2)
	stream.resultJSONChan <- ""
	stream.resultJSONChan <- `{"choices":[]}`

	got, err := stream.RecvJSON()
	if err != nil || got != `{"choices":[]}` {
		t.Fatalf("RecvJSON() = (%q, %v), want nonempty chunk and nil error", got, err)
	}
}

func TestRecvJSONCompletionDuringReceive(t *testing.T) {
	workerErr := errors.New("worker unavailable")
	for _, tc := range []struct {
		name      string
		chunks    []string
		workerErr error
		wantErr   error
	}{
		{
			name:    "completed_output",
			chunks:  []string{`{"choices":[{"delta":{"content":"Hello"}}]}`, `{"usage":{"completion_tokens":1}}`},
			wantErr: io.EOF,
		},
		{name: "worker_error", workerErr: workerErr, wantErr: workerErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			observedCtx := &receiveObservedContext{Context: ctx, observed: make(chan struct{})}
			stream := newRecvJSONTestStream(observedCtx, len(tc.chunks))
			type receiveResult struct {
				chunks []string
				err    error
			}
			done := make(chan receiveResult, 1)
			go func() {
				var result receiveResult
				for {
					chunk, err := stream.RecvJSON()
					if err != nil {
						result.err = err
						done <- result
						return
					}
					result.chunks = append(result.chunks, chunk)
				}
			}()

			// Wait for the receiver to observe the context while the stream is
			// still open and empty, then complete it without timing sleeps.
			select {
			case <-observedCtx.observed:
			case <-time.After(5 * time.Second):
				t.Fatal("receiver did not start waiting for a response")
			}
			for _, chunk := range tc.chunks {
				stream.resultJSONChan <- chunk
			}
			if tc.workerErr != nil {
				stream.errChan <- tc.workerErr
			}
			close(stream.resultJSONChan)
			close(stream.errChan)
			cancel()

			select {
			case got := <-done:
				if !slices.Equal(got.chunks, tc.chunks) || !errors.Is(got.err, tc.wantErr) {
					t.Fatalf("received (%q, %v), want (%q, %v)", got.chunks, got.err, tc.chunks, tc.wantErr)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("receiver did not finish after stream completion")
			}
		})
	}
}

func TestRecvJSONPreservesBackendError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := newRecvJSONTestStream(ctx, 1)
	wantErr := errors.New("worker unavailable")
	stream.errChan <- wantErr
	close(stream.resultJSONChan)
	close(stream.errChan)
	cancel()

	if got, err := stream.RecvJSON(); got != "" || !errors.Is(err, wantErr) {
		t.Fatalf("RecvJSON() = (%q, %v), want (\"\", %v)", got, err, wantErr)
	}
}

func TestRecvJSONContextErrorWhileStreamOpen(t *testing.T) {
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	deadlineCtx, deadlineCancel := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer deadlineCancel()

	for _, tc := range []struct {
		name string
		ctx  context.Context
		want error
	}{
		{name: "canceled", ctx: canceledCtx, want: context.Canceled},
		{name: "deadline", ctx: deadlineCtx, want: context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := newRecvJSONTestStream(tc.ctx, 1)
			if got, err := stream.RecvJSON(); got != "" || !errors.Is(err, tc.want) {
				t.Fatalf("RecvJSON() = (%q, %v), want (\"\", %v)", got, err, tc.want)
			}
		})
	}
}

func newRecvJSONTestStream(ctx context.Context, resultBufferSize int) *GrpcChatCompletionStream {
	return &GrpcChatCompletionStream{
		ctx:            ctx,
		resultJSONChan: make(chan string, resultBufferSize),
		errChan:        make(chan error, 1),
	}
}

type receiveObservedContext struct {
	context.Context
	observed chan struct{}
	once     sync.Once
}

func (ctx *receiveObservedContext) Done() <-chan struct{} {
	ctx.once.Do(func() { close(ctx.observed) })
	return ctx.Context.Done()
}
