package smg

import "sort"

// ToolCallDelta is one piece of a tool call in a streaming response. The
// first delta of an index carries the call's id, its type and the function's
// name; every later delta of that index carries a further piece of the
// arguments, with no id and no name.
type ToolCallDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function FunctionCallDelta `json:"function"`
}

// FunctionCallDelta is the function part of a tool-call delta: the name with
// the first delta of a call, a piece of the arguments with every delta that
// carries one. A field the delta does not carry is omitted, so a delta
// re-marshalled by a consumer keeps the shape the converter sent.
type FunctionCallDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// toolCallAssembler rebuilds the tool calls a stream delivers in pieces, by
// index, the way the non-streaming answer of the gateway's own API is built:
// one call per index, its id, type and name from the first delta that carries
// them, its arguments the concatenation of every delta's piece.
type toolCallAssembler struct {
	calls map[int]*ToolCall
}

// add takes the tool-call deltas of one streamed choice delta.
func (a *toolCallAssembler) add(deltas []ToolCallDelta) {
	for _, delta := range deltas {
		if a.calls == nil {
			a.calls = make(map[int]*ToolCall)
		}
		call, ok := a.calls[delta.Index]
		if !ok {
			call = &ToolCall{}
			a.calls[delta.Index] = call
		}
		if delta.ID != "" {
			call.ID = delta.ID
		}
		if delta.Type != "" {
			call.Type = delta.Type
		}
		if delta.Function.Name != "" {
			call.Function.Name = delta.Function.Name
		}
		call.Function.Arguments += delta.Function.Arguments
	}
}

// toolCalls returns the assembled calls in index order; nil when the stream
// carried none.
func (a *toolCallAssembler) toolCalls() []ToolCall {
	if len(a.calls) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(a.calls))
	for index := range a.calls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	calls := make([]ToolCall, 0, len(indexes))
	for _, index := range indexes {
		calls = append(calls, *a.calls[index])
	}
	return calls
}
