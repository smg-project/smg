package smg

import (
	"encoding/json"
	"strings"
	"testing"
)

// A tool declared on a system message travels with the message, next to the
// request-level tools, for the provider profiles that define such tools.
func TestMessageDeclaredToolsRoundTrip(t *testing.T) {
	message := ChatMessage{
		Role:    "system",
		Content: "",
		Tools:   []Tool{{Type: "function", Function: Function{Name: "get_weather", Parameters: map[string]interface{}{"type": "object"}}}},
	}
	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(encoded), `"tools":[{"type":"function","function":{"name":"get_weather"`) {
		t.Fatalf("message-declared tools missing from %s", encoded)
	}
	var decoded ChatMessage
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(decoded.Tools) != 1 || decoded.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("decoded %+v", decoded)
	}
}

// A message without tools carries no tools field.
func TestMessageWithoutToolsOmitsTheField(t *testing.T) {
	encoded, err := json.Marshal(ChatMessage{Role: "user", Content: "hi"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(encoded), `"tools"`) {
		t.Fatalf("unexpected tools field in %s", encoded)
	}
}
