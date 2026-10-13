package smg

import (
	"encoding/json"
	"reflect"
	"testing"
)

// The streaming converter sends one tool call in pieces: the id, the type and
// the name with the first delta of its index, the arguments split over the
// deltas that follow (no id, no name). The non-streaming path assembles one
// call from them.
func TestToolCallStreamedInPiecesAssemblesIntoOneCall(t *testing.T) {
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}`,
	}
	var calls toolCallAssembler
	for _, chunkJSON := range chunks {
		var chunk ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(chunkJSON), &chunk); err != nil {
			t.Fatalf("chunk: %v", err)
		}
		for _, choice := range chunk.Choices {
			calls.add(choice.Delta.ToolCalls)
		}
	}
	want := []ToolCall{{ID: "call_1", Type: "function", Function: FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}}}
	if got := calls.toolCalls(); !reflect.DeepEqual(got, want) {
		t.Fatalf("assembled %+v, want %+v", got, want)
	}
}

// Two calls whose pieces interleave come back as two calls, in index order,
// each with its own arguments.
func TestInterleavedToolCallsAssembleByIndex(t *testing.T) {
	var calls toolCallAssembler
	calls.add([]ToolCallDelta{{Index: 1, ID: "call_2", Type: "function", Function: FunctionCall{Name: "get_time"}}})
	calls.add([]ToolCallDelta{{Index: 0, ID: "call_1", Type: "function", Function: FunctionCall{Name: "get_weather"}}})
	calls.add([]ToolCallDelta{{Index: 1, Function: FunctionCall{Arguments: `{"zone":`}}})
	calls.add([]ToolCallDelta{{Index: 0, Function: FunctionCall{Arguments: `{"city":"Paris"}`}}})
	calls.add([]ToolCallDelta{{Index: 1, Function: FunctionCall{Arguments: `"UTC"}`}}})
	want := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
		{ID: "call_2", Type: "function", Function: FunctionCall{Name: "get_time", Arguments: `{"zone":"UTC"}`}},
	}
	if got := calls.toolCalls(); !reflect.DeepEqual(got, want) {
		t.Fatalf("assembled %+v, want %+v", got, want)
	}
}

// A stream without tool calls leaves the message's tool calls unset.
func TestNoToolCallDeltasLeaveToolCallsUnset(t *testing.T) {
	var calls toolCallAssembler
	calls.add(nil)
	if got := calls.toolCalls(); got != nil {
		t.Fatalf("got %+v, want nil", got)
	}
}
