package connect

import (
        "encoding/json"
        "net/http"
)

type modelInfo struct {
        ID      string `json:"id"`
        Object  string `json:"object"`
        Created int64  `json:"created"`
        OwnedBy string `json:"owned_by"`
}

type modelsResponse struct {
        Object string      `json:"object"`
        Data   []modelInfo `json:"data"`
}

// Models returns the list of available models (OpenAI-compatible).
//
//encore:api public raw path=/v1/models
func Models(w http.ResponseWriter, req *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Header().Set("Access-Control-Allow-Origin", "*")

        resp := modelsResponse{
                Object: "list",
                Data: []modelInfo{
                        {
                                ID:      displayModel,
                                Object:  "model",
                                Created: 1714000000,
                                OwnedBy: "together",
                        },
                },
        }

        json.NewEncoder(w).Encode(resp)
}
