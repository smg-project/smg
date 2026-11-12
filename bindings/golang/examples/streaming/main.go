// Streaming example demonstrating real-time streaming with SMG Go SDK
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

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

	// Create streaming chat completion request
	req := smg.ChatCompletionRequest{
		Model: "default",
		Messages: []smg.ChatMessage{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: "Write a short poem about spring",
			},
		},
		Stream:              true,
		Temperature:         float32Ptr(0.7),
		MaxCompletionTokens: intPtr(500),
		SkipSpecialTokens:   true,
		Tools:               nil, // Use nil instead of empty slice to avoid template errors
	}

	// Create streaming completion
	ctx := context.Background()
	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		log.Fatalf("Failed to create stream: %v", err)
	}
	defer stream.Close()

	fmt.Println("=== Streaming Response ===")
	fmt.Println()

	var fullContent strings.Builder
	chunkCount := 0
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false

	for {
		jsonStr, err := stream.RecvJSON()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}

		// Parse the JSON response
		var chunk smg.ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			log.Printf("Failed to parse chunk: %v", err)
			continue
		}

		chunkCount++

		// Extract content from delta
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fmt.Print(choice.Delta.Content)
				fullContent.WriteString(choice.Delta.Content)

				// Track first token time (TTFT)
				if !firstTokenReceived {
					firstTokenTime = time.Now()
					firstTokenReceived = true
					ttft := firstTokenTime.Sub(startTime)
					fmt.Printf("\n[TTFT: %v]\n", ttft)
				}
			}

			if choice.FinishReason != "" {
				fmt.Printf("\n\n[Finished: %s]\n", choice.FinishReason)
			}
		}
	}

	// Calculate metrics
	if firstTokenReceived {
		elapsed := time.Since(startTime)
		tokensPerSecond := float64(fullContent.Len()) / elapsed.Seconds()
		fmt.Printf("\n=== Metrics ===\n")
		fmt.Printf("Total chunks: %d\n", chunkCount)
		fmt.Printf("Total content length: %d characters\n", fullContent.Len())
		fmt.Printf("Time elapsed: %v\n", elapsed)
		fmt.Printf("Tokens per second: %.2f\n", tokensPerSecond)
	}
}

func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
