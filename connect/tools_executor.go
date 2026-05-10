package connect

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const maxAgenticIterations = 10

// agenticResult holds the final text content after the full tool-calling loop.
type agenticResult struct {
	content          string
	reasoningContent string
}

// runAgenticLoop calls Together AI, executes any tool calls server-side, feeds
// results back, and repeats until the model returns plain text (or the iteration
// limit is reached).  It returns the final text content.
func runAgenticLoop(
	ctx context.Context,
	messages []ollamaMessage,
	tools []toolDef,
	apiKey string,
	baseURL string,
	model string,
	onToolCall func(name, argsJSON, result string), // optional progress callback
) agenticResult {
	client := &http.Client{Transport: &http.Transport{DisableCompression: true}}

	for i := 0; i < maxAgenticIterations; i++ {
		payload := togetherRequest{
			Model:     model,
			Messages:  messages,
			Stream:    false,
			Reasoning: reasoningConfig{Enabled: false},
			Tools:     tools,
		}
		payloadBytes, _ := json.Marshal(payload)

		apiReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL, bytes.NewReader(payloadBytes))
		if err != nil {
			return agenticResult{content: fmt.Sprintf(`{"error":"%s"}`, err.Error())}
		}
		apiReq.Header.Set("Content-Type", "application/json")
		apiReq.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := client.Do(apiReq)
		if err != nil {
			return agenticResult{content: fmt.Sprintf(`{"error":"%s"}`, err.Error())}
		}
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var nr nonStreamResponse
		if err := json.Unmarshal(respBody, &nr); err != nil || len(nr.Choices) == 0 {
			return agenticResult{content: `{"error":"bad upstream response"}`}
		}

		msg := nr.Choices[0].Message

		// No tool calls → final answer
		if len(msg.ToolCalls) == 0 {
			content := ""
			reasoning := ""
			if msg.Content != nil {
				content = stripThinkTags(*msg.Content)
			}
			if msg.ReasoningContent != nil {
				reasoning = *msg.ReasoningContent
			}
			return agenticResult{content: content, reasoningContent: reasoning}
		}

		// Build the assistant message that contains the tool_calls.
		created := time.Now().Unix()
		assistantTCs := buildToolCalls(msg.ToolCalls, created)

		assistantMsg := ollamaMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: assistantTCs,
		}
		messages = append(messages, assistantMsg)

		// Execute each tool and append a tool-result message.
		for _, tc := range assistantTCs {
			argsStr := argsToString(tc.Function.Arguments)
			result := executeTool(tc.Function.Name, argsStr)

			if onToolCall != nil {
				onToolCall(tc.Function.Name, argsStr, result)
			}

			messages = append(messages, ollamaMessage{
				Role:    "tool",
				Content: result,
			})
		}
	}

	return agenticResult{content: `{"error":"max agentic iterations reached"}`}
}

// doNonStreamWithTools performs a full server-side agentic loop and writes
// the final OpenAI-compatible JSON response.
func doNonStreamWithTools(
	ctx context.Context,
	w http.ResponseWriter,
	messages []ollamaMessage,
	tools []toolDef,
	incomingID string,
) {
	result := runAgenticLoop(ctx, messages, tools, togetherAPIKey, togetherBaseURL, togetherModel, nil)

	created := time.Now().Unix()
	content := result.content
	var reasoning *string
	if result.reasoningContent != "" {
		reasoning = strPtr(result.reasoningContent)
	}

	nr := nonStreamResponse{
		ID:      incomingID,
		Object:  "chat.completion",
		Created: created,
		Model:   displayModel,
		Choices: []nonStreamChoice{
			{
				Index: 0,
				Message: openAIResponseMessage{
					Role:             "assistant",
					Content:          strPtr(content),
					ReasoningContent: reasoning,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(nr)
}

// doStreamWithTools performs a full server-side agentic loop and then streams
// the final text back to the client as SSE chunks.
func doStreamWithTools(
	ctx context.Context,
	w http.ResponseWriter,
	messages []ollamaMessage,
	tools []toolDef,
	writeSSE func(d delta, finishReason *string),
	flush func(),
	id string,
) {
	result := runAgenticLoop(ctx, messages, tools, togetherAPIKey, togetherBaseURL, togetherModel, nil)

	if result.content != "" {
		writeSSE(delta{Content: strPtr(result.content)}, nil)
	}
	if result.reasoningContent != "" {
		writeSSE(delta{ReasoningContent: result.reasoningContent}, nil)
	}

	stop := "stop"
	writeSSE(delta{}, &stop)
	flush()
}
