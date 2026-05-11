package responses

import (
	"fmt"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_FlattensNamespaceTools(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"input": "please use MCP tools",
		"tools": [
			{
				"type": "namespace",
				"name": "mcp__test_mcp__",
				"tools": [
					{
						"type": "function",
						"name": "add_numbers",
						"description": "Add two numbers together",
						"parameters": {
							"type": "object",
							"properties": {
								"a": {"type": "number"},
								"b": {"type": "number"}
							},
							"required": ["a", "b"]
						}
					},
					{
						"type": "function",
						"name": "echo",
						"description": "Echo a message",
						"parameters": {
							"type": "object",
							"properties": {
								"message": {"type": "string"}
							},
							"required": ["message"]
						}
					}
				]
			},
			{
				"type": "web_search",
				"name": "web_search_preview"
			},
			{
				"type": "function",
				"name": "list_files",
				"description": "List files",
				"parameters": {
					"type": "object",
					"properties": {
						"path": {"type": "string"}
					}
				}
			}
		],
		"tool_choice": {
			"type": "function",
			"function": {
				"name": "echo"
			}
		}
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("tools.#").Int(); got != 3 {
		t.Fatalf("expected 3 flattened tools, got %d: %s", got, string(out))
	}

	wantNames := []string{
		"mcp__test_mcp__add_numbers",
		"mcp__test_mcp__echo",
		"list_files",
	}
	for idx, want := range wantNames {
		path := fmt.Sprintf("tools.%d.function.name", idx)
		if got := result.Get(path).String(); got != want {
			t.Fatalf("%s = %q, want %q. Output: %s", path, got, want, string(out))
		}
	}

	if got := result.Get("tools.0.function.description").String(); got != "Add two numbers together" {
		t.Fatalf("unexpected description for first tool: %q", got)
	}
	if got := result.Get("tools.0.function.parameters.required.0").String(); got != "a" {
		t.Fatalf("unexpected first required field: %q", got)
	}
	if got := result.Get("tools.0.function.parameters.required.1").String(); got != "b" {
		t.Fatalf("unexpected second required field: %q", got)
	}

	if got := result.Get("tool_choice.function.name").String(); got != "mcp__test_mcp__echo" {
		t.Fatalf("tool_choice.function.name = %q, want %q. Output: %s", got, "mcp__test_mcp__echo", string(out))
	}
	if got := result.Get("tool_choice.type").String(); got != "function" {
		t.Fatalf("tool_choice.type = %q, want %q. Output: %s", got, "function", string(out))
	}
	if result.Get("tools.3").Exists() {
		t.Fatalf("unexpected extra tool emitted: %s", result.Get("tools.3").Raw)
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_IgnoresUnsupportedBuiltinsWhileKeepingNamespaceChildren(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"input": "mix namespace and builtins",
		"tools": [
			{
				"type": "web_search",
				"name": "web_search"
			},
			{
				"type": "namespace",
				"name": "mcp__test_mcp__",
				"tools": [
					{
						"type": "function",
						"name": "get_time",
						"parameters": {"type": "object", "properties": {}}
					}
				]
			}
		],
		"tool_choice": "auto"
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("tools.#").Int(); got != 1 {
		t.Fatalf("expected only namespace child to remain, got %d tools: %s", got, string(out))
	}
	if got := result.Get("tools.0.function.name").String(); got != "mcp__test_mcp__get_time" {
		t.Fatalf("tools.0.function.name = %q, want %q", got, "mcp__test_mcp__get_time")
	}
	if got := result.Get("tool_choice").String(); got != "auto" {
		t.Fatalf("tool_choice = %q, want %q", got, "auto")
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_SanitizesStrictFunctionNames(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"input": [
			{
				"type": "function_call",
				"call_id": "call_report",
				"name": "mcp__redqueen__.redqueen_session_report_assessment",
				"arguments": "{}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_report",
				"output": "ok"
			}
		],
		"tools": [
			{
				"type": "function",
				"name": "mcp__redqueen__.redqueen_session_report_assessment",
				"parameters": {"type": "object", "properties": {}}
			},
			{
				"type": "namespace",
				"name": "mcp__ops__",
				"tools": [
					{
						"type": "function",
						"name": "analyze.security/logs",
						"parameters": {"type": "object", "properties": {}}
					}
				]
			}
		],
		"tool_choice": {
			"type": "function",
			"function": {
				"name": "mcp__redqueen__.redqueen_session_report_assessment"
			}
		}
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("tools.0.function.name").String(); got != "mcp__redqueen___redqueen_session_report_assessment" {
		t.Fatalf("tools.0.function.name = %q, want %q. Output: %s", got, "mcp__redqueen___redqueen_session_report_assessment", string(out))
	}
	if got := result.Get("tools.1.function.name").String(); got != "mcp__ops__analyze_security_logs" {
		t.Fatalf("tools.1.function.name = %q, want %q. Output: %s", got, "mcp__ops__analyze_security_logs", string(out))
	}
	if got := result.Get("tool_choice.function.name").String(); got != "mcp__redqueen___redqueen_session_report_assessment" {
		t.Fatalf("tool_choice.function.name = %q, want %q. Output: %s", got, "mcp__redqueen___redqueen_session_report_assessment", string(out))
	}
	if got := result.Get("messages.0.tool_calls.0.function.name").String(); got != "mcp__redqueen___redqueen_session_report_assessment" {
		t.Fatalf("messages.0.tool_calls.0.function.name = %q, want %q. Output: %s", got, "mcp__redqueen___redqueen_session_report_assessment", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_PreservesExplicitEnableThinking(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"input": "hello",
		"enable_thinking": true
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("MiniMax-M2.5", input, false)
	result := gjson.ParseBytes(out)

	if !result.Get("enable_thinking").Exists() {
		t.Fatalf("expected enable_thinking to be preserved. Output: %s", string(out))
	}
	if got := result.Get("enable_thinking").Bool(); !got {
		t.Fatalf("enable_thinking = %v, want true. Output: %s", got, string(out))
	}
	if result.Get("thinking").Exists() {
		t.Fatalf("thinking should not be defaulted when enable_thinking is explicit. Output: %s", string(out))
	}
	if got := result.Get("messages.0.role").String(); got != "user" {
		t.Fatalf("messages.0.role = %q, want %q. Output: %s", got, "user", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_AttachesStandaloneReasoningToToolCallAssistantMessage(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"input": [
			{
				"type": "reasoning",
				"summary": [
					{
						"type": "summary_text",
						"text": "Need to inspect the missing dependency first."
					}
				]
			},
			{
				"type": "function_call",
				"call_id": "call_exec",
				"name": "exec_command",
				"arguments": "{\"cmd\":\"python3 solve.py\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_exec",
				"output": "Traceback..."
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("messages.0.role").String(); got != "assistant" {
		t.Fatalf("messages.0.role = %q, want %q. Output: %s", got, "assistant", string(out))
	}
	if got := result.Get("messages.0.reasoning_content").String(); got != "Need to inspect the missing dependency first." {
		t.Fatalf("messages.0.reasoning_content = %q, want %q. Output: %s", got, "Need to inspect the missing dependency first.", string(out))
	}
	if got := result.Get("messages.0.tool_calls.0.function.name").String(); got != "exec_command" {
		t.Fatalf("messages.0.tool_calls.0.function.name = %q, want %q. Output: %s", got, "exec_command", string(out))
	}
	if got := result.Get("messages.1.role").String(); got != "tool" {
		t.Fatalf("messages.1.role = %q, want %q. Output: %s", got, "tool", string(out))
	}
	if got := result.Get("messages.1.tool_call_id").String(); got != "call_exec" {
		t.Fatalf("messages.1.tool_call_id = %q, want %q. Output: %s", got, "call_exec", string(out))
	}
	if result.Get("messages.1.reasoning_content").Exists() {
		t.Fatalf("tool messages must not carry reasoning_content. Output: %s", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_AttachesStandaloneReasoningAfterUserMessageToToolCallAssistantMessage(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"enable_thinking": true,
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": "Please thinking test"
					}
				]
			},
			{
				"type": "reasoning",
				"summary": [
					{
						"type": "summary_text",
						"text": "Let me check what Python packages are available first."
					}
				]
			},
			{
				"type": "function_call",
				"call_id": "call_00_3iQijZAjxVCVEPJtWt8X0087",
				"name": "exec_command",
				"arguments": "{\"cmd\":\"python3 solve.py\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_00_3iQijZAjxVCVEPJtWt8X0087",
				"output": "ok"
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("messages.0.role").String(); got != "user" {
		t.Fatalf("messages.0.role = %q, want %q. Output: %s", got, "user", string(out))
	}
	if got := result.Get("messages.0.content.0.text").String(); got != "Please thinking test" {
		t.Fatalf("messages.0.content.0.text = %q, want %q. Output: %s", got, "Please thinking test", string(out))
	}
	if got := result.Get("messages.1.role").String(); got != "assistant" {
		t.Fatalf("messages.1.role = %q, want %q. Output: %s", got, "assistant", string(out))
	}
	if got := result.Get("messages.1.reasoning_content").String(); got != "Let me check what Python packages are available first." {
		t.Fatalf("messages.1.reasoning_content = %q, want %q. Output: %s", got, "Let me check what Python packages are available first.", string(out))
	}
	if got := result.Get("messages.1.tool_calls.0.id").String(); got != "call_00_3iQijZAjxVCVEPJtWt8X0087" {
		t.Fatalf("messages.1.tool_calls.0.id = %q, want %q. Output: %s", got, "call_00_3iQijZAjxVCVEPJtWt8X0087", string(out))
	}
	if got := result.Get("messages.1.tool_calls.0.function.name").String(); got != "exec_command" {
		t.Fatalf("messages.1.tool_calls.0.function.name = %q, want %q. Output: %s", got, "exec_command", string(out))
	}
	if got := result.Get("messages.2.role").String(); got != "tool" {
		t.Fatalf("messages.2.role = %q, want %q. Output: %s", got, "tool", string(out))
	}
	if got := result.Get("messages.2.tool_call_id").String(); got != "call_00_3iQijZAjxVCVEPJtWt8X0087" {
		t.Fatalf("messages.2.tool_call_id = %q, want %q. Output: %s", got, "call_00_3iQijZAjxVCVEPJtWt8X0087", string(out))
	}
	if result.Get("messages.2.reasoning_content").Exists() {
		t.Fatalf("tool messages must not carry reasoning_content. Output: %s", string(out))
	}
	if got := result.Get("enable_thinking").Bool(); !got {
		t.Fatalf("enable_thinking = %v, want true. Output: %s", got, string(out))
	}
	if result.Get("thinking").Exists() {
		t.Fatalf("thinking should not be defaulted when enable_thinking is explicit. Output: %s", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_MergesAssistantTextWithFollowingToolCall(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"enable_thinking": true,
		"input": [
			{
				"type": "reasoning",
				"summary": [
					{
						"type": "summary_text",
						"text": "Good, I have the challenge context. Let me extract and inspect the attachment."
					}
				]
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [
					{
						"type": "output_text",
						"text": "Let me extract the zip file and inspect its contents."
					}
				]
			},
			{
				"type": "function_call",
				"call_id": "call_extract",
				"name": "exec_command",
				"arguments": "{\"cmd\":\"unzip -l challenge.zip\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_extract",
				"output": "output.txt\ntask.py"
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("messages.0.role").String(); got != "assistant" {
		t.Fatalf("messages.0.role = %q, want %q. Output: %s", got, "assistant", string(out))
	}
	if got := result.Get("messages.0.reasoning_content").String(); got != "Good, I have the challenge context. Let me extract and inspect the attachment." {
		t.Fatalf("messages.0.reasoning_content = %q, want %q. Output: %s", got, "Good, I have the challenge context. Let me extract and inspect the attachment.", string(out))
	}
	if got := result.Get("messages.0.content.0.text").String(); got != "Let me extract the zip file and inspect its contents." {
		t.Fatalf("messages.0.content.0.text = %q, want %q. Output: %s", got, "Let me extract the zip file and inspect its contents.", string(out))
	}
	if got := result.Get("messages.0.tool_calls.0.id").String(); got != "call_extract" {
		t.Fatalf("messages.0.tool_calls.0.id = %q, want %q. Output: %s", got, "call_extract", string(out))
	}
	if got := result.Get("messages.0.tool_calls.0.function.name").String(); got != "exec_command" {
		t.Fatalf("messages.0.tool_calls.0.function.name = %q, want %q. Output: %s", got, "exec_command", string(out))
	}
	if got := result.Get("messages.1.role").String(); got != "tool" {
		t.Fatalf("messages.1.role = %q, want %q. Output: %s", got, "tool", string(out))
	}
	if result.Get("messages.2").Exists() {
		t.Fatalf("expected assistant text and tool_call to merge into one assistant message. Output: %s", string(out))
	}
	if got := result.Get("enable_thinking").Bool(); !got {
		t.Fatalf("enable_thinking = %v, want true. Output: %s", got, string(out))
	}
	if result.Get("thinking").Exists() {
		t.Fatalf("thinking should not be defaulted when enable_thinking is explicit. Output: %s", string(out))
	}
	if got := result.Get("messages.1.tool_call_id").String(); got != "call_extract" {
		t.Fatalf("messages.1.tool_call_id = %q, want %q. Output: %s", got, "call_extract", string(out))
	}
	if result.Get("messages.1.reasoning_content").Exists() {
		t.Fatalf("tool messages must not carry reasoning_content. Output: %s", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletions_FlushesCompletedToolGroupBeforeNextReasoningTurn(t *testing.T) {
	t.Parallel()

	input := []byte(`{
		"enable_thinking": true,
		"input": [
			{
				"type": "function_call",
				"call_id": "call_prev",
				"name": "previous_tool",
				"arguments": "{}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_prev",
				"output": "ok"
			},
			{
				"type": "message",
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": "Please thinking test"
					}
				]
			},
			{
				"type": "reasoning",
				"summary": [
					{
						"type": "summary_text",
						"text": "Let me quickly check the attachment before submitting the assessment."
					}
				]
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [
					{
						"type": "output_text",
						"text": "Let me quickly check the attachment before submitting the assessment."
					}
				]
			},
			{
				"type": "function_call",
				"call_id": "call_next",
				"name": "exec_command",
				"arguments": "{\"cmd\":\"cat /workspace/attachments/Power tower.py\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_next",
				"output": "from Crypto.Util.number import *"
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("deepseek-v4-flash", input, false)
	result := gjson.ParseBytes(out)

	if got := result.Get("messages.0.role").String(); got != "assistant" {
		t.Fatalf("messages.0.role = %q, want %q. Output: %s", got, "assistant", string(out))
	}
	if got := result.Get("messages.0.tool_calls.0.id").String(); got != "call_prev" {
		t.Fatalf("messages.0.tool_calls.0.id = %q, want %q. Output: %s", got, "call_prev", string(out))
	}
	if result.Get("messages.0.reasoning_content").Exists() {
		t.Fatalf("previous tool-call assistant must not steal next-turn reasoning_content. Output: %s", string(out))
	}
	if got := result.Get("messages.1.role").String(); got != "tool" {
		t.Fatalf("messages.1.role = %q, want %q. Output: %s", got, "tool", string(out))
	}
	if got := result.Get("messages.2.role").String(); got != "user" {
		t.Fatalf("messages.2.role = %q, want %q. Output: %s", got, "user", string(out))
	}
	if got := result.Get("messages.3.role").String(); got != "assistant" {
		t.Fatalf("messages.3.role = %q, want %q. Output: %s", got, "assistant", string(out))
	}
	if got := result.Get("messages.3.reasoning_content").String(); got != "Let me quickly check the attachment before submitting the assessment." {
		t.Fatalf("messages.3.reasoning_content = %q, want %q. Output: %s", got, "Let me quickly check the attachment before submitting the assessment.", string(out))
	}
	if got := result.Get("messages.3.content.0.text").String(); got != "Let me quickly check the attachment before submitting the assessment." {
		t.Fatalf("messages.3.content.0.text = %q, want %q. Output: %s", got, "Let me quickly check the attachment before submitting the assessment.", string(out))
	}
	if got := result.Get("messages.3.tool_calls.0.id").String(); got != "call_next" {
		t.Fatalf("messages.3.tool_calls.0.id = %q, want %q. Output: %s", got, "call_next", string(out))
	}
	if got := result.Get("messages.4.role").String(); got != "tool" {
		t.Fatalf("messages.4.role = %q, want %q. Output: %s", got, "tool", string(out))
	}
}

func TestQualifyResponsesNamespaceToolName(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		namespace string
		child     string
		want      string
	}{
		{
			name:      "appends separator when namespace lacks suffix",
			namespace: "mcp__test_mcp",
			child:     "echo",
			want:      "mcp__test_mcp__echo",
		},
		{
			name:      "reuses trailing separator",
			namespace: "mcp__test_mcp__",
			child:     "echo",
			want:      "mcp__test_mcp__echo",
		},
		{
			name:      "keeps already qualified child",
			namespace: "mcp__test_mcp__",
			child:     "mcp__test_mcp__echo",
			want:      "mcp__test_mcp__echo",
		},
		{
			name:      "keeps other mcp child untouched",
			namespace: "mcp__test_mcp__",
			child:     "mcp__other__echo",
			want:      "mcp__other__echo",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := qualifyResponsesNamespaceToolName(tt.namespace, tt.child); got != tt.want {
				t.Fatalf("qualifyResponsesNamespaceToolName(%q, %q) = %q, want %q", tt.namespace, tt.child, got, tt.want)
			}
		})
	}
}
