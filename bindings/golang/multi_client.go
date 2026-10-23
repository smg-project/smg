// Package smg provides a Go SDK for SMG (Shepherd Model Gateway) gRPC API.
//
// This file provides the MultiClient for load-balanced multi-worker deployments.
package smg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/lightseek/smg/go-grpc-sdk/internal/ffi"
)

// MultiClient is a client that distributes requests across multiple gRPC workers
// using a configurable load balancing policy.
//
// Thread-safe: All public methods are safe for concurrent use.
type MultiClient struct {
	endpoints     string
	tokenizerPath string
	policyName    string
	ffiClient     *ffi.MultiWorkerClientHandle
	mu            sync.RWMutex
}

// MultiClientConfig holds configuration for creating a new multi-worker client.
type MultiClientConfig struct {
	// Endpoints is a comma-separated list of gRPC endpoint URLs
	// (e.g., "grpc://host1:20000,grpc://host2:20001,grpc://host3:20002")
	// Required field. Each endpoint must include the scheme (grpc://) and port number.
	Endpoints string

	// TokenizerPath is the path to the tokenizer directory containing
	// tokenizer configuration files (e.g., tokenizer.json, vocab.json).
	// Required field.
	TokenizerPath string

	// PolicyName is the load balancing policy to use.
	// Available policies: "round_robin", "random", "cache_aware"
	// Defaults to "round_robin" if not specified.
	PolicyName string
}

// NewMultiClient creates a new multi-worker client with load balancing.
//
// The client maintains connections to multiple gRPC workers and distributes
// requests using the configured policy. Call Close() to release resources.
//
// Returns an error if:
// - Endpoints is empty
// - TokenizerPath is empty
// - Connection to any worker fails
// - Invalid policy name is specified
func NewMultiClient(config MultiClientConfig) (*MultiClient, error) {
	if config.Endpoints == "" {
		return nil, errors.New("endpoints is required")
	}
	if config.TokenizerPath == "" {
		return nil, errors.New("tokenizer path is required")
	}

	policyName := config.PolicyName
	if policyName == "" {
		policyName = "round_robin"
	}

	ffiClient, err := ffi.NewMultiWorkerClient(config.Endpoints, config.TokenizerPath, policyName)
	if err != nil {
		return nil, fmt.Errorf("failed to create multi-worker client: %w", err)
	}

	return &MultiClient{
		endpoints:     config.Endpoints,
		tokenizerPath: config.TokenizerPath,
		policyName:    policyName,
		ffiClient:     ffiClient,
	}, nil
}

// Close closes the client and releases all resources.
//
// After Close() is called, the client cannot be used for further requests.
// Calling Close() multiple times is safe and idempotent.
func (c *MultiClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.ffiClient != nil {
		c.ffiClient.Free()
		c.ffiClient = nil
	}
	return nil
}

// WorkerCount returns the total number of workers configured.
func (c *MultiClient) WorkerCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.ffiClient == nil {
		return 0
	}
	return c.ffiClient.WorkerCount()
}

// HealthyWorkerCount returns the number of currently healthy workers.
func (c *MultiClient) HealthyWorkerCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.ffiClient == nil {
		return 0
	}
	return c.ffiClient.HealthyCount()
}

// SetWorkerHealth marks a worker as healthy or unhealthy by index.
// This is useful for implementing external health checking.
func (c *MultiClient) SetWorkerHealth(workerIndex int, healthy bool) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.ffiClient == nil {
		return errors.New("client is closed")
	}
	return c.ffiClient.SetWorkerHealth(workerIndex, healthy)
}

// PolicyName returns the name of the configured load balancing policy.
func (c *MultiClient) PolicyName() string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.ffiClient == nil {
		return ""
	}
	return c.ffiClient.PolicyName()
}

// CreateChatCompletion creates a non-streaming chat completion with context support.
//
// Context Support:
// The ctx parameter is fully supported for cancellation and timeouts.
//
// Note: Internally, this creates a stream and collects all chunks,
// so context monitoring happens at the chunk level.
func (c *MultiClient) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	// For non-streaming, we'll collect all chunks and return the final response
	req.Stream = true

	if len(req.Tools) == 0 {
		req.Tools = nil
	}

	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var fullContent strings.Builder
	var toolCalls toolCallAssembler
	var finishReason string
	var usage Usage
	var responseID string
	var created int64
	var model string
	var systemFingerprint string

	for {
		chunkJSON, err := stream.RecvJSON()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		var chunk ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(chunkJSON), &chunk); err != nil {
			return nil, fmt.Errorf("failed to parse chunk: %w", err)
		}

		if chunk.ID != "" {
			responseID = chunk.ID
		}
		if chunk.Created > 0 {
			created = chunk.Created
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.SystemFingerprint != "" {
			systemFingerprint = chunk.SystemFingerprint
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fullContent.WriteString(choice.Delta.Content)
			}
			toolCalls.add(choice.Delta.ToolCalls)
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}

	message := Message{
		Role:      "assistant",
		Content:   fullContent.String(),
		ToolCalls: toolCalls.toolCalls(),
	}

	if finishReason == "" {
		finishReason = "stop"
	}

	return &ChatCompletionResponse{
		ID:                responseID,
		Object:            "chat.completion",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices: []Choice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}, nil
}

// MultiClientStream represents a streaming chat completion from a multi-worker client
type MultiClientStream struct {
	ffiStream *ffi.SglangStreamHandle
	ctx       context.Context
	cancel    context.CancelFunc
}

func (s *MultiClientStream) RecvJSON() (string, error) {
	// Check context first
	select {
	case <-s.ctx.Done():
		return "", s.ctx.Err()
	default:
	}

	responseJSON, isDone, err := s.ffiStream.ReadNext()
	if err != nil {
		return "", err
	}
	if isDone {
		return "", io.EOF
	}
	return responseJSON, nil
}

// Close closes the stream and cancels any pending operations.
func (s *MultiClientStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.ffiStream != nil {
		s.ffiStream.Free()
		s.ffiStream = nil
	}
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion with load balancing.
//
// The request is routed to a healthy worker using the configured load balancing policy.
func (c *MultiClient) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*MultiClientStream, error) {
	c.mu.RLock()
	ffiClient := c.ffiClient
	c.mu.RUnlock()

	if ffiClient == nil {
		return nil, errors.New("multi-worker client is closed")
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqJSON, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request to map: %w", err)
	}

	if _, exists := reqMap["tools"]; !exists {
		reqMap["tools"] = []interface{}{}
	}

	reqJSON, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request map to JSON: %w", err)
	}

	ffiStream, err := ffiClient.ChatCompletionStream(string(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create stream: %w", err)
	}

	streamCtx, cancel := context.WithCancel(ctx)
	return &MultiClientStream{
		ffiStream: ffiStream,
		ctx:       streamCtx,
		cancel:    cancel,
	}, nil
}
