package service

import (
	"context"
	"fmt"
	"strings"

	smg "github.com/smg-project/smg/bindings/golang"
)

// ChatClient interface defines methods for chat completion operations.
// Both smg.Client and smg.MultiClient implement this interface.
type ChatClient interface {
	CreateChatCompletion(ctx context.Context, req smg.ChatCompletionRequest) (*smg.ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req smg.ChatCompletionRequest) (ChatStream, error)
	Close() error
}

// ChatStream interface defines methods for streaming chat completion.
type ChatStream interface {
	RecvJSON() (string, error)
	Close() error
}

// singleClientWrapper wraps *smg.Client to implement ChatClient interface
type singleClientWrapper struct {
	client *smg.Client
}

func (w *singleClientWrapper) CreateChatCompletion(ctx context.Context, req smg.ChatCompletionRequest) (*smg.ChatCompletionResponse, error) {
	return w.client.CreateChatCompletion(ctx, req)
}

func (w *singleClientWrapper) CreateChatCompletionStream(ctx context.Context, req smg.ChatCompletionRequest) (ChatStream, error) {
	return w.client.CreateChatCompletionStream(ctx, req)
}

func (w *singleClientWrapper) Close() error {
	return w.client.Close()
}

// multiClientWrapper wraps *smg.MultiClient to implement ChatClient interface
type multiClientWrapper struct {
	client *smg.MultiClient
}

func (w *multiClientWrapper) CreateChatCompletion(ctx context.Context, req smg.ChatCompletionRequest) (*smg.ChatCompletionResponse, error) {
	return w.client.CreateChatCompletion(ctx, req)
}

func (w *multiClientWrapper) CreateChatCompletionStream(ctx context.Context, req smg.ChatCompletionRequest) (ChatStream, error) {
	return w.client.CreateChatCompletionStream(ctx, req)
}

func (w *multiClientWrapper) Close() error {
	return w.client.Close()
}

// SMGService wraps SMG client (supports both single and multi-worker)
type SMGService struct {
	chatClient ChatClient
	// Keep references for info purposes
	isMultiWorker bool
	workerCount   int
	policyName    string
}

// NewSMGService creates a new SMG service.
// If endpoints contains multiple comma-separated endpoints, uses MultiClient with load balancing.
// Otherwise uses single Client for backwards compatibility.
func NewSMGService(endpoints, tokenizerPath, policyName string) (*SMGService, error) {
	// Parse endpoints
	endpointList := strings.Split(endpoints, ",")
	for i := range endpointList {
		endpointList[i] = strings.TrimSpace(endpointList[i])
	}

	// Filter empty endpoints
	var validEndpoints []string
	for _, ep := range endpointList {
		if ep != "" {
			validEndpoints = append(validEndpoints, ep)
		}
	}

	if len(validEndpoints) == 0 {
		return nil, fmt.Errorf("no valid gRPC endpoints provided in endpoints string: %q", endpoints)
	}

	if len(validEndpoints) > 1 {
		// Multiple endpoints: use MultiClient with load balancing
		multiClient, err := smg.NewMultiClient(smg.MultiClientConfig{
			Endpoints:     strings.Join(validEndpoints, ","),
			TokenizerPath: tokenizerPath,
			PolicyName:    policyName,
		})
		if err != nil {
			return nil, err
		}
		return &SMGService{
			chatClient:    &multiClientWrapper{client: multiClient},
			isMultiWorker: true,
			workerCount:   multiClient.WorkerCount(),
			policyName:    multiClient.PolicyName(),
		}, nil
	}

	// Single endpoint: use regular Client for backwards compatibility
	client, err := smg.NewClient(smg.ClientConfig{
		Endpoint:      validEndpoints[0],
		TokenizerPath: tokenizerPath,
	})
	if err != nil {
		return nil, err
	}

	return &SMGService{
		chatClient:    &singleClientWrapper{client: client},
		isMultiWorker: false,
		workerCount:   1,
		policyName:    "",
	}, nil
}

// ChatClient returns the underlying chat client interface
func (s *SMGService) ChatClient() ChatClient {
	return s.chatClient
}

// IsMultiWorker returns true if using multi-worker setup
func (s *SMGService) IsMultiWorker() bool {
	return s.isMultiWorker
}

// WorkerCount returns the number of workers
func (s *SMGService) WorkerCount() int {
	return s.workerCount
}

// PolicyName returns the load balancing policy name (empty for single worker)
func (s *SMGService) PolicyName() string {
	return s.policyName
}

// Close closes the SMG client
func (s *SMGService) Close() error {
	if s.chatClient != nil {
		return s.chatClient.Close()
	}
	return nil
}
