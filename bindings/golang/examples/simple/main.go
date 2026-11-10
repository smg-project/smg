// Simple example demonstrating basic usage of SMG Go SDK
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	smg "github.com/smg-project/smg/bindings/golang"
)

func main() {
	// Get configuration from environment or command line
	endpoint := os.Getenv("SGL_GRPC_ENDPOINT")
	if endpoint == "" {
		endpoint = "grpc://localhost:20000"
	}

	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "./examples/tokenizer"
	}

	// Create client
	client, err := smg.NewClient(smg.ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create chat completion request
	req := smg.ChatCompletionRequest{
		Model: "default",
		Messages: []smg.ChatMessage{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: "Write a short poem about summer",
			},
		},
		Stream:              false,
		Temperature:         float32Ptr(0.7),
		MaxCompletionTokens: intPtr(200),
		SkipSpecialTokens:   true,
		Tools:               nil, // Use nil instead of empty slice to avoid template errors
	}

	// Create completion
	ctx := context.Background()
	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		log.Fatalf("Failed to create completion: %v", err)
	}

	// Print response
	fmt.Println("=== Response ===")
	fmt.Printf("ID: %s\n", resp.ID)
	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Created: %d\n", resp.Created)
	fmt.Println("\nContent:")
	for _, choice := range resp.Choices {
		fmt.Println(choice.Message.Content)
	}
	fmt.Printf("\nFinish Reason: %s\n", resp.Choices[0].FinishReason)
	fmt.Printf("\nUsage: Prompt=%d, Completion=%d, Total=%d\n",
		resp.Usage.PromptTokens,
		resp.Usage.CompletionTokens,
		resp.Usage.TotalTokens,
	)
}

func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
