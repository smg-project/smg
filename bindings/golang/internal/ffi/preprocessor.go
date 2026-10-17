// Package ffi provides Go bindings for SMG's Rust FFI (Foreign Function Interface).
package ffi

/*
#cgo LDFLAGS: -lsmg_go -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes (must match client.go)
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Preprocessor functions
SglErrorCode sgl_preprocess_chat_request(
    const char* request_json,
    const char* tokenizer_path,
    char** prompt_text_out,
    uint32_t** token_ids_out,
    size_t* token_ids_len_out,
    char** tool_constraints_json_out,
    char** tools_json_out,
    int32_t* skip_special_tokens_out,
    int32_t* prompt_tokens_out,
    char** error_out
);

// Opaque handle (must match grpc_converter.go)
typedef void* TokenizerHandle;

SglErrorCode sgl_preprocess_chat_request_with_tokenizer(
    const char* request_json,
    void* tokenizer_handle,
    char** prompt_text_out,
    uint32_t** token_ids_out,
    size_t* token_ids_len_out,
    char** tool_constraints_json_out,
    char** tools_json_out,
    int32_t* skip_special_tokens_out,
    int32_t* prompt_tokens_out,
    char** error_out
);

SglErrorCode sgl_chat_requires_reasoning_with_tokenizer(
    const char* request_json,
    void* tokenizer_handle,
    int32_t* require_reasoning_out,
    char** error_out
);

void sgl_preprocessed_request_free(
    char* prompt_text,
    uint32_t* token_ids,
    size_t token_ids_len,
    char* tool_constraints_json,
    char* tools_json
);

// Memory management
void sgl_free_string(char* s);
void sgl_free_token_ids(uint32_t* ptr, size_t count);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// PreprocessedRequest represents a preprocessed chat request
type PreprocessedRequest struct {
	PromptText          string
	TokenIDs            []uint32
	ToolConstraintsJSON string
	// ToolsJSON is the tool list the response parser resolves calls against:
	// the request-level tools and the tools declared on messages, as the
	// preprocessor resolved them ("" when the request declares none).
	ToolsJSON string
	// SkipSpecialTokens is whether the engine and the response converter skip
	// special tokens: the request's setting, unless the response parser needs
	// the engine's tool-call markers kept (tools under a choice that allows
	// calls, no JSON-schema constraint), as the preprocessor derived it.
	SkipSpecialTokens bool
	PromptTokens      int32
	// Internal pointers for memory management
	promptTextPtr          *C.char
	tokenIDsPtr            *C.uint32_t
	tokenIDsLen            uintptr
	toolConstraintsJSONPtr *C.char
	toolsJSONPtr           *C.char
}

// PreprocessChatRequest preprocesses a chat completion request
//
// This function:
// 1. Applies chat_template to messages
// 2. Tokenizes the processed text
// 3. Generates tool constraints (if tools are present)
//
// Returns the preprocessed request data and any error.
func PreprocessChatRequest(requestJSON, tokenizerPath string) (*PreprocessedRequest, error) {
	requestJSONC := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(requestJSONC))

	tokenizerPathC := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(tokenizerPathC))

	var promptTextOut *C.char
	var tokenIDsOut *C.uint32_t
	var tokenIDsLenOut C.size_t
	var toolConstraintsJSONOut *C.char
	var toolsJSONOut *C.char
	var skipSpecialTokensOut C.int32_t
	var promptTokensOut C.int32_t
	var errorOut *C.char

	errorCode := C.sgl_preprocess_chat_request(
		requestJSONC,
		tokenizerPathC,
		&promptTextOut,
		&tokenIDsOut,
		&tokenIDsLenOut,
		&toolConstraintsJSONOut,
		&toolsJSONOut,
		&skipSpecialTokensOut,
		&promptTokensOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return nil, fmt.Errorf("preprocessing failed: %s", errorMsg)
	}

	result := &PreprocessedRequest{
		PromptText:          C.GoString(promptTextOut),
		TokenIDs:            make([]uint32, tokenIDsLenOut),
		ToolConstraintsJSON: "",
		PromptTokens:        int32(promptTokensOut),
	}

	// Copy token IDs
	if tokenIDsOut != nil && tokenIDsLenOut > 0 {
		tokenIDsSlice := (*[1 << 30]C.uint32_t)(unsafe.Pointer(tokenIDsOut))[:tokenIDsLenOut:tokenIDsLenOut]
		for i := range result.TokenIDs {
			result.TokenIDs[i] = uint32(tokenIDsSlice[i])
		}
	}

	// Copy tool constraints JSON if present
	if toolConstraintsJSONOut != nil {
		result.ToolConstraintsJSON = C.GoString(toolConstraintsJSONOut)
	}

	// Copy the response parser's tool list if present
	if toolsJSONOut != nil {
		result.ToolsJSON = C.GoString(toolsJSONOut)
	}

	result.SkipSpecialTokens = skipSpecialTokensOut != 0

	// Store pointers for later cleanup
	result.promptTextPtr = promptTextOut
	result.tokenIDsPtr = tokenIDsOut
	result.tokenIDsLen = uintptr(tokenIDsLenOut)
	result.toolConstraintsJSONPtr = toolConstraintsJSONOut
	result.toolsJSONPtr = toolsJSONOut

	return result, nil
}

// PreprocessChatRequestWithTokenizer preprocesses a chat completion request using an existing tokenizer handle
//
// This function is similar to PreprocessChatRequest, but accepts a TokenizerHandle
// instead of creating a new tokenizer. This allows reusing a cached tokenizer instance,
// significantly reducing initialization overhead in concurrent scenarios.
//
// Returns the preprocessed request data and any error.
func PreprocessChatRequestWithTokenizer(requestJSON string, tokenizerHandle *TokenizerHandle) (*PreprocessedRequest, error) {
	requestJSONC := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(requestJSONC))

	if tokenizerHandle == nil || tokenizerHandle.handle == nil {
		return nil, fmt.Errorf("invalid tokenizer handle")
	}

	var promptTextOut *C.char
	var tokenIDsOut *C.uint32_t
	var tokenIDsLenOut C.size_t
	var toolConstraintsJSONOut *C.char
	var toolsJSONOut *C.char
	var skipSpecialTokensOut C.int32_t
	var promptTokensOut C.int32_t
	var errorOut *C.char

	errorCode := C.sgl_preprocess_chat_request_with_tokenizer(
		requestJSONC,
		unsafe.Pointer(tokenizerHandle.handle), // Convert *C.TokenizerHandle to void*
		&promptTextOut,
		&tokenIDsOut,
		&tokenIDsLenOut,
		&toolConstraintsJSONOut,
		&toolsJSONOut,
		&skipSpecialTokensOut,
		&promptTokensOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return nil, fmt.Errorf("preprocessing failed: %s", errorMsg)
	}

	result := &PreprocessedRequest{
		PromptText:          C.GoString(promptTextOut),
		TokenIDs:            make([]uint32, tokenIDsLenOut),
		ToolConstraintsJSON: "",
		PromptTokens:        int32(promptTokensOut),
	}

	// Copy token IDs
	if tokenIDsOut != nil && tokenIDsLenOut > 0 {
		tokenIDsSlice := (*[1 << 30]C.uint32_t)(unsafe.Pointer(tokenIDsOut))[:tokenIDsLenOut:tokenIDsLenOut]
		for i := range result.TokenIDs {
			result.TokenIDs[i] = uint32(tokenIDsSlice[i])
		}
	}

	// Copy tool constraints JSON if present
	if toolConstraintsJSONOut != nil {
		result.ToolConstraintsJSON = C.GoString(toolConstraintsJSONOut)
	}

	// Copy the response parser's tool list if present
	if toolsJSONOut != nil {
		result.ToolsJSON = C.GoString(toolsJSONOut)
	}

	result.SkipSpecialTokens = skipSpecialTokensOut != 0

	// Store pointers for later cleanup
	result.promptTextPtr = promptTextOut
	result.tokenIDsPtr = tokenIDsOut
	result.tokenIDsLen = uintptr(tokenIDsLenOut)
	result.toolConstraintsJSONPtr = toolConstraintsJSONOut
	result.toolsJSONPtr = toolsJSONOut

	return result, nil
}

// ChatRequiresReasoningWithTokenizer returns whether the request should ask
// SGLang to count reasoning tokens, using the tokenizer's thinking defaults.
func ChatRequiresReasoningWithTokenizer(requestJSON string, tokenizerHandle *TokenizerHandle) (bool, error) {
	requestJSONC := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(requestJSONC))

	if tokenizerHandle == nil || tokenizerHandle.handle == nil {
		return false, fmt.Errorf("invalid tokenizer handle")
	}

	var requireReasoningOut C.int32_t
	var errorOut *C.char

	errorCode := C.sgl_chat_requires_reasoning_with_tokenizer(
		requestJSONC,
		unsafe.Pointer(tokenizerHandle.handle),
		&requireReasoningOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return false, fmt.Errorf("failed to determine require_reasoning: %s", errorMsg)
	}

	return requireReasoningOut != 0, nil
}

// Free frees the memory allocated for a preprocessed request
func (p *PreprocessedRequest) Free() {
	if p.promptTextPtr != nil || p.tokenIDsPtr != nil || p.toolConstraintsJSONPtr != nil || p.toolsJSONPtr != nil {
		C.sgl_preprocessed_request_free(
			p.promptTextPtr,
			p.tokenIDsPtr,
			C.size_t(p.tokenIDsLen),
			p.toolConstraintsJSONPtr,
			p.toolsJSONPtr,
		)
		// Clear pointers to prevent double-free
		p.promptTextPtr = nil
		p.tokenIDsPtr = nil
		p.tokenIDsLen = 0
		p.toolConstraintsJSONPtr = nil
		p.toolsJSONPtr = nil
	}
}

// FreePreprocessedRequest frees the memory allocated for a preprocessed request
// This is a convenience function for direct pointer management
func FreePreprocessedRequest(promptTextPtr *C.char, tokenIDsPtr *C.uint32_t, tokenIDsLen uintptr, toolConstraintsJSONPtr *C.char, toolsJSONPtr *C.char) {
	if promptTextPtr != nil || tokenIDsPtr != nil || toolConstraintsJSONPtr != nil || toolsJSONPtr != nil {
		C.sgl_preprocessed_request_free(
			promptTextPtr,
			tokenIDsPtr,
			C.size_t(tokenIDsLen),
			toolConstraintsJSONPtr,
			toolsJSONPtr,
		)
	}
}
