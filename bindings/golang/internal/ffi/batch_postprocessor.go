// Package ffi provides Go bindings for SGLang's Rust FFI (Foreign Function Interface).
package ffi

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ErrPostprocessorClosed is the error of a chunk or a flush that arrives after
// Close freed the converter.
var ErrPostprocessorClosed = errors.New("invalid converter handle: the postprocessor is closed")

// BatchPostprocessor handles batch postprocessing of stream chunks to reduce FFI overhead
type BatchPostprocessor struct {
	mu            sync.Mutex // Protects concurrent access to the FFI converter
	converter     *GrpcResponseConverterHandle
	buffer        []string
	batchSize     int
	flushInterval time.Duration
	lastFlush     time.Time
	timer         *time.Timer
}

// NewBatchPostprocessor creates a new batch postprocessor
func NewBatchPostprocessor(converter *GrpcResponseConverterHandle, batchSize int, flushInterval time.Duration) *BatchPostprocessor {
	if batchSize <= 0 {
		batchSize = 1
	}
	if flushInterval < 0 {
		flushInterval = 0
	}

	return &BatchPostprocessor{
		converter:     converter,
		buffer:        make([]string, 0, batchSize),
		batchSize:     batchSize,
		flushInterval: flushInterval,
		lastFlush:     time.Now(),
	}
}

// AddChunk adds a chunk to the buffer and processes if batch is full
func (b *BatchPostprocessor) AddChunk(chunkJSON string) (results []string, shouldFlush bool, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.converter == nil {
		return nil, false, ErrPostprocessorClosed
	}
	if b.batchSize == 1 {
		openaiJSON, _, err := PostprocessStreamChunk(b.converter, chunkJSON)
		if err != nil {
			return nil, false, err
		}
		return []string{openaiJSON}, false, nil
	}

	b.buffer = append(b.buffer, chunkJSON)
	shouldProcess := len(b.buffer) >= b.batchSize
	shouldFlushTimeout := b.flushInterval > 0 && time.Since(b.lastFlush) >= b.flushInterval

	if shouldProcess || shouldFlushTimeout {
		return b.processBatchLocked()
	}

	return nil, false, nil
}

// Flush processes any remaining chunks in the buffer
func (b *BatchPostprocessor) Flush() (results []string, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(b.buffer) == 0 {
		return nil, nil
	}
	if b.converter == nil {
		return nil, ErrPostprocessorClosed
	}

	res, _, err := b.processBatchLocked()
	return res, err
}

// Close frees the converter under the lock that every chunk and flush takes,
// so a chunk still inside the converter finishes first and a later one gets
// ErrPostprocessorClosed instead of a freed converter. A second Close frees
// nothing more.
func (b *BatchPostprocessor) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.converter != nil {
		FreeGrpcResponseConverter(b.converter)
		b.converter = nil
	}
	b.buffer = b.buffer[:0]
	if b.timer != nil {
		b.timer.Stop()
		b.timer = nil
	}
}

// processBatchLocked processes the current buffer and returns results
// Must be called with b.mu held
func (b *BatchPostprocessor) processBatchLocked() (results []string, shouldFlush bool, err error) {
	if len(b.buffer) == 0 {
		return nil, false, nil
	}

	var sb strings.Builder
	sb.Grow(len(b.buffer) * 200)
	sb.WriteString(`[`)
	for i, chunkJSONStr := range b.buffer {
		if i > 0 {
			sb.WriteString(`,`)
		}
		sb.WriteString(chunkJSONStr)
	}
	sb.WriteString(`]`)
	bufferJSON := sb.String()

	resultJSON, _, err := PostprocessStreamChunksBatch(
		b.converter,
		bufferJSON,
		b.batchSize*2,
	)
	if err != nil {
		return nil, false, fmt.Errorf("batch postprocessing failed: %w", err)
	}

	var resultArray []json.RawMessage
	if err := json.Unmarshal([]byte(resultJSON), &resultArray); err != nil {
		return nil, false, fmt.Errorf("failed to unmarshal results array: %w", err)
	}

	resultStrings := make([]string, 0, len(resultArray))
	for _, rawMsg := range resultArray {
		resultStrings = append(resultStrings, string(rawMsg))
	}

	b.buffer = b.buffer[:0]
	b.lastFlush = time.Now()

	if b.timer != nil {
		b.timer.Stop()
		b.timer = nil
	}

	return resultStrings, false, nil
}

// Reset clears the buffer and resets the postprocessor state
func (b *BatchPostprocessor) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.buffer = b.buffer[:0]
	b.lastFlush = time.Now()
	if b.timer != nil {
		b.timer.Stop()
		b.timer = nil
	}
}
