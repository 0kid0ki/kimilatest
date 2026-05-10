// Service connect implements an OpenAI-compatible SSE streaming endpoint backed by Ollama API.
package connect

import (
        "bufio"
        "bytes"
        "encoding/json"
        "fmt"
        "io"
        "net/http"
        "regexp"
        "strings"
        "time"
)

const togetherAPIKey = "tgp_v1_ZrHkCuh1ndrbEgryMEq0AcMO3C6WrMtRPsHs6QTOJDQ"
const togetherBaseURL = "https://api.together.xyz/v1/chat/completions"
const togetherModel = "deepseek-ai/DeepSeek-V3.1"
const displayModel = "DeepSeek-V3.1"

var thinkTagRe = regexp.MustCompile(`(?s)<think>.*?</think>`)

// stripThinkTags removes <think>...</think> blocks from content.
func stripThinkTags(s string) string {
        return strings.TrimSpace(thinkTagRe.ReplaceAllString(s, ""))
}

// ── Types ─────────────────────────────────────────────────────────────────────

type toolFunction struct {
        Name      string `json:"name"`
        Arguments any    `json:"arguments,omitempty"`
}

type toolCall struct {
        ID       string       `json:"id,omitempty"`
        Type     string       `json:"type,omitempty"`
        Function toolFunction `json:"function"`
}

type incomingMessage struct {
        Role       string     `json:"role"`
        Content    any        `json:"content"`
        ToolCalls  []toolCall `json:"tool_calls,omitempty"`
        ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ollamaMessage struct {
        Role      string     `json:"role"`
        Content   string     `json:"content"`
        Thinking  string     `json:"thinking,omitempty"`
        ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

type toolDef struct {
        Type     string      `json:"type"`
        Function toolFuncDef `json:"function"`
}

type toolFuncDef struct {
        Name        string `json:"name"`
        Description string `json:"description,omitempty"`
        Parameters  any    `json:"parameters,omitempty"`
}

type togetherRequest struct {
        Model     string          `json:"model"`
        Messages  []ollamaMessage `json:"messages"`
        Stream    bool            `json:"stream"`
        Reasoning reasoningConfig `json:"reasoning"`
        Tools     []toolDef       `json:"tools,omitempty"`
}

type reasoningConfig struct {
        Enabled bool `json:"enabled"`
}

type ollamaChunk struct {
        Message ollamaMessage `json:"message"`
        Done    bool          `json:"done"`
}

type streamingToolCall struct {
        Index    int          `json:"index"`
        ID       string       `json:"id,omitempty"`
        Type     string       `json:"type,omitempty"`
        Function toolFunction `json:"function"`
}

type delta struct {
        Role             string              `json:"role,omitempty"`
        Content          *string             `json:"content,omitempty"`
        ReasoningContent string              `json:"reasoning_content,omitempty"`
        ToolCalls        []streamingToolCall `json:"tool_calls,omitempty"`
}

type chunkChoice struct {
        Index        int     `json:"index"`
        Delta        delta   `json:"delta"`
        FinishReason *string `json:"finish_reason"`
}

type openAIChunk struct {
        ID      string        `json:"id"`
        Object  string        `json:"object"`
        Created int64         `json:"created"`
        Model   string        `json:"model"`
        Choices []chunkChoice `json:"choices"`
}

type incomingRequest struct {
        Messages   []incomingMessage `json:"messages"`
        Tools      []toolDef         `json:"tools,omitempty"`
        ToolChoice any               `json:"tool_choice,omitempty"`
        Stream     *bool             `json:"stream,omitempty"`
}

// openAIResponseMessage is the OpenAI-compatible message format for non-streaming responses.
// Content is a pointer so it can be null when tool_calls are present.
type openAIResponseMessage struct {
        Role             string     `json:"role"`
        Content          *string    `json:"content"`
        ReasoningContent *string    `json:"reasoning_content,omitempty"`
        ToolCalls        []toolCall `json:"tool_calls,omitempty"`
}

type nonStreamChoice struct {
        Index        int                   `json:"index"`
        Message      openAIResponseMessage `json:"message"`
        FinishReason string                `json:"finish_reason"`
}

type nonStreamResponse struct {
        ID      string            `json:"id"`
        Object  string            `json:"object"`
        Created int64             `json:"created"`
        Model   string            `json:"model"`
        Choices []nonStreamChoice `json:"choices"`
        Usage   struct {
                PromptTokens     int `json:"prompt_tokens"`
                CompletionTokens int `json:"completion_tokens"`
                TotalTokens      int `json:"total_tokens"`
        } `json:"usage"`
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func contentString(c any) string {
        if c == nil {
                return ""
        }
        if s, ok := c.(string); ok {
                return s
        }
        b, _ := json.Marshal(c)
        return string(b)
}

func strPtr(s string) *string { return &s }

func buildTogetherMessages(inc []incomingMessage) []ollamaMessage {
        sysContent := systemPrompt
        for _, m := range inc {
                if strings.ToLower(m.Role) == "system" {
                        if s := contentString(m.Content); s != "" {
                                sysContent = s
                        }
                        break
                }
        }

        out := []ollamaMessage{{Role: "system", Content: sysContent}}
        for _, m := range inc {
                role := strings.ToLower(m.Role)
                if role == "system" {
                        continue
                }

                if role == "tool" || m.ToolCallID != "" {
                        out = append(out, ollamaMessage{
                                Role:    "tool",
                                Content: contentString(m.Content),
                        })
                        continue
                }

                om := ollamaMessage{
                        Role:    m.Role,
                        Content: contentString(m.Content),
                }

                if len(m.ToolCalls) > 0 {
                        tcs := make([]toolCall, 0, len(m.ToolCalls))
                        for _, tc := range m.ToolCalls {
                                otc := toolCall{
                                        ID:   tc.ID,
                                        Type: tc.Type,
                                        Function: toolFunction{
                                                Name: tc.Function.Name,
                                        },
                                }
                                switch v := tc.Function.Arguments.(type) {
                                case string:
                                        var obj any
                                        if json.Unmarshal([]byte(v), &obj) == nil {
                                                otc.Function.Arguments = obj
                                        } else {
                                                otc.Function.Arguments = v
                                        }
                                default:
                                        otc.Function.Arguments = v
                                }
                                tcs = append(tcs, otc)
                        }
                        om.ToolCalls = tcs
                }

                out = append(out, om)
        }
        return out
}

func argsToString(v any) string {
        switch s := v.(type) {
        case string:
                return s
        case nil:
                return "{}"
        default:
                b, _ := json.Marshal(s)
                return string(b)
        }
}

// buildToolCalls converts Ollama tool calls to OpenAI format (arguments as JSON string).
func buildToolCalls(src []toolCall, created int64) []toolCall {
        tcs := make([]toolCall, 0, len(src))
        for i, tc := range src {
                tcs = append(tcs, toolCall{
                        ID:   fmt.Sprintf("call_%d_%d", created, i),
                        Type: "function",
                        Function: toolFunction{
                                Name:      tc.Function.Name,
                                Arguments: argsToString(tc.Function.Arguments),
                        },
                })
        }
        return tcs
}

// ── Handler ───────────────────────────────────────────────────────────────────

// Connect is an OpenAI-compatible SSE streaming chat completions endpoint.
//
//encore:api public raw path=/v1/chat/completions
func Connect(w http.ResponseWriter, req *http.Request) {
        if req.Method == http.MethodOptions {
                w.Header().Set("Access-Control-Allow-Origin", "*")
                w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
                w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
                w.WriteHeader(http.StatusNoContent)
                return
        }
        if req.Method != http.MethodPost {
                http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
                return
        }

        body, err := io.ReadAll(req.Body)
        if err != nil {
                http.Error(w, "Bad Request", http.StatusBadRequest)
                return
        }

        var incoming incomingRequest
        if err := json.Unmarshal(body, &incoming); err != nil {
                http.Error(w, "Invalid JSON", http.StatusBadRequest)
                return
        }

        allTools := append(incoming.Tools, getRegisteredToolDefs()...)
        messages := buildTogetherMessages(incoming.Messages)
        hasTools := len(allTools) > 0
        wantStream := incoming.Stream == nil || *incoming.Stream

        // ── Non-streaming mode (stream:false) ─────────────────────────────────────
        if !wantStream {
                if hasTools {
                        doNonStreamWithTools(req.Context(), w, messages, allTools, fmt.Sprintf("chatcmpl-%d", time.Now().Unix()))
                        return
                }

                payload := togetherRequest{
                        Model:     togetherModel,
                        Messages:  messages,
                        Stream:    false,
                        Reasoning: reasoningConfig{Enabled: false},
                }
                payloadBytes, _ := json.Marshal(payload)
                apiReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, togetherBaseURL, bytes.NewReader(payloadBytes))
                if err != nil {
                        http.Error(w, "Internal Server Error", http.StatusInternalServerError)
                        return
                }
                apiReq.Header.Set("Content-Type", "application/json")
                apiReq.Header.Set("Authorization", "Bearer "+togetherAPIKey)
                resp, err := (&http.Client{Transport: &http.Transport{DisableCompression: true}}).Do(apiReq)
                if err != nil {
                        http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                        return
                }
                defer resp.Body.Close()

                respBody, _ := io.ReadAll(resp.Body)
                var togetherResp nonStreamResponse
                json.Unmarshal(respBody, &togetherResp)

                if len(togetherResp.Choices) == 0 {
                        http.Error(w, "Empty response from upstream", http.StatusBadGateway)
                        return
                }

                created := togetherResp.Created
                if created == 0 {
                        created = time.Now().Unix()
                }
                finishReason := togetherResp.Choices[0].FinishReason
                msg := togetherResp.Choices[0].Message

                if msg.Content != nil {
                        content := stripThinkTags(*msg.Content)
                        msg.Content = &content
                }

                nr := nonStreamResponse{
                        ID:      togetherResp.ID,
                        Object:  "chat.completion",
                        Created: created,
                        Model:   displayModel,
                        Choices: []nonStreamChoice{{Index: 0, Message: msg, FinishReason: finishReason}},
                        Usage:   togetherResp.Usage,
                }
                w.Header().Set("Content-Type", "application/json")
                w.Header().Set("Access-Control-Allow-Origin", "*")
                json.NewEncoder(w).Encode(nr)
                return
        }

        // ── Streaming mode ────────────────────────────────────────────────────────
        w.Header().Set("Content-Type", "text/event-stream")
        w.Header().Set("Cache-Control", "no-cache, no-transform")
        w.Header().Set("Connection", "keep-alive")
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("X-Accel-Buffering", "no")
        w.WriteHeader(http.StatusOK)

        flusher, canFlush := w.(http.Flusher)
        created := time.Now().Unix()
        id := fmt.Sprintf("chatcmpl-%d", created)

        writeSSE := func(d delta, finishReason *string) {
                chunk := openAIChunk{
                        ID:      id,
                        Object:  "chat.completion.chunk",
                        Created: created,
                        Model:   displayModel,
                        Choices: []chunkChoice{{Index: 0, Delta: d, FinishReason: finishReason}},
                }
                b, _ := json.Marshal(chunk)
                fmt.Fprintf(w, "data: %s\n\n", b)
                if canFlush {
                        flusher.Flush()
                }
        }

        flush := func() {
                fmt.Fprintf(w, "data: [DONE]\n\n")
                if canFlush {
                        flusher.Flush()
                }
        }

        doRequest := func(stream bool) (*http.Response, error) {
                payload := togetherRequest{
                        Model:     togetherModel,
                        Messages:  messages,
                        Stream:    stream,
                        Reasoning: reasoningConfig{Enabled: false},
                        Tools:     allTools,
                }
                payloadBytes, _ := json.Marshal(payload)
                apiReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, togetherBaseURL, bytes.NewReader(payloadBytes))
                if err != nil {
                        return nil, err
                }
                apiReq.Header.Set("Content-Type", "application/json")
                apiReq.Header.Set("Authorization", "Bearer "+togetherAPIKey)
                return (&http.Client{Transport: &http.Transport{DisableCompression: true}}).Do(apiReq)
        }

        writeSSE(delta{Role: "assistant"}, nil)

        if hasTools {
                // Run full server-side agentic loop: execute tools, feed results back,
                // repeat until the model returns plain text, then stream the final answer.
                doStreamWithTools(req.Context(), w, messages, allTools, writeSSE, flush, id)
                return
        }

        // No tools — pure real-time streaming
        resp, err := doRequest(true)
        if err != nil {
                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                if canFlush {
                        flusher.Flush()
                }
                return
        }
        defer resp.Body.Close()

        reader := bufio.NewReader(resp.Body)
        for {
                line, readErr := reader.ReadString('\n')
                line = strings.TrimSpace(line)

                if strings.HasPrefix(line, "data: ") {
                        data := strings.TrimPrefix(line, "data: ")
                        if data == "[DONE]" {
                                flush()
                                return
                        }

                        var chunk openAIChunk
                        if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
                                choice := chunk.Choices[0]
                                if choice.FinishReason != nil {
                                        writeSSE(delta{}, choice.FinishReason)
                                        flush()
                                        return
                                }
                                writeSSE(choice.Delta, nil)
                        }
                }

                if readErr == io.EOF || readErr != nil {
                        break
                }
        }
        flush()
}
