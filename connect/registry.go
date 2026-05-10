package connect

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// RegisteredTool — narzędzie skonfigurowane przez użytkownika
type RegisteredTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	URL         string         `json:"url"`
	Headers     map[string]string `json:"headers,omitempty"`
	CreatedAt   int64          `json:"created_at"`
}

var (
	toolRegistry   = map[string]*RegisteredTool{}
	registryMu     sync.RWMutex
)

// ── API: POST /v1/tools — rejestracja narzędzia ───────────────────────────────
//
//encore:api public raw path=/v1/tools
func ToolsAPI(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if req.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	switch req.Method {
	case http.MethodGet:
		listTools(w, req)
	case http.MethodPost:
		registerTool(w, req)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

// ── API: DELETE /v1/tools/:name ───────────────────────────────────────────────
//
//encore:api public raw path=/v1/tools/:name
func ToolByName(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if req.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	name := strings.TrimPrefix(req.URL.Path, "/v1/tools/")

	switch req.Method {
	case http.MethodGet:
		getTool(w, name)
	case http.MethodDelete:
		deleteTool(w, name)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

func listTools(w http.ResponseWriter, _ *http.Request) {
	registryMu.RLock()
	defer registryMu.RUnlock()

	tools := make([]*RegisteredTool, 0, len(toolRegistry))
	for _, t := range toolRegistry {
		tools = append(tools, t)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   tools,
	})
}

func getTool(w http.ResponseWriter, name string) {
	registryMu.RLock()
	t, ok := toolRegistry[name]
	registryMu.RUnlock()

	if !ok {
		http.Error(w, `{"error":"tool not found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(t)
}

func registerTool(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
		return
	}

	var t RegisteredTool
	if err := json.Unmarshal(body, &t); err != nil || t.Name == "" || t.URL == "" {
		http.Error(w, `{"error":"name and url are required"}`, http.StatusBadRequest)
		return
	}
	t.CreatedAt = time.Now().Unix()

	registryMu.Lock()
	toolRegistry[t.Name] = &t
	registryMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(t)
}

func deleteTool(w http.ResponseWriter, name string) {
	registryMu.Lock()
	_, ok := toolRegistry[name]
	if ok {
		delete(toolRegistry, name)
	}
	registryMu.Unlock()

	if !ok {
		http.Error(w, `{"error":"tool not found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"deleted": name})
}

// ── Execution ─────────────────────────────────────────────────────────────────

// executeTool — wywołuje URL narzędzia z argumentami, zwraca wynik jako string
func executeTool(name string, argsJSON string) string {
	registryMu.RLock()
	t, ok := toolRegistry[name]
	registryMu.RUnlock()

	if !ok {
		return fmt.Sprintf(`{"error":"tool '%s' not registered"}`, name)
	}

	var args map[string]any
	json.Unmarshal([]byte(argsJSON), &args)

	payload := map[string]any{
		"name":      name,
		"arguments": args,
	}
	payloadBytes, _ := json.Marshal(payload)

	client := &http.Client{Timeout: 30 * time.Second}
	httpReq, err := http.NewRequest(http.MethodPost, t.URL, bytes.NewReader(payloadBytes))
	if err != nil {
		return fmt.Sprintf(`{"error":"%s"}`, err.Error())
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range t.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Sprintf(`{"error":"%s"}`, err.Error())
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32768))
	return strings.TrimSpace(string(body))
}

// getRegisteredToolDefs — zwraca zarejestrowane narzędzia w formacie Ollama
func getRegisteredToolDefs() []toolDef {
	registryMu.RLock()
	defer registryMu.RUnlock()

	defs := make([]toolDef, 0, len(toolRegistry))
	for _, t := range toolRegistry {
		defs = append(defs, toolDef{
			Type: "function",
			Function: toolFuncDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return defs
}
