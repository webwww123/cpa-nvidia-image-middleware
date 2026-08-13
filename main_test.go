package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

func newTestService(t *testing.T, upstream http.HandlerFunc) *imageService {
	t.Helper()
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)
	service, err := newImageService(server.URL, 5*time.Second, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	return service
}

func TestNewImageServiceWithProxyRejectsInvalidProxy(t *testing.T) {
	if _, err := newImageServiceWithProxy("https://example.com", "://bad-proxy", 5*time.Second, nil); err == nil {
		t.Fatal("expected invalid proxy URL error")
	}
}

func TestNewImageServiceWithProxyConfiguresTransport(t *testing.T) {
	service, err := newImageServiceWithProxy("https://example.com", "http://127.0.0.1:18080", 5*time.Second, nil)
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := service.client.Transport.(*http.Transport)
	if !ok || transport.Proxy == nil {
		t.Fatal("expected upstream proxy transport")
	}
	request := httptest.NewRequest(http.MethodGet, "https://ai.api.nvidia.com/v1/test", nil)
	proxyURL, err := transport.Proxy(request)
	if err != nil {
		t.Fatal(err)
	}
	if proxyURL.String() != "http://127.0.0.1:18080" {
		t.Fatalf("proxy URL = %q", proxyURL.String())
	}
}

func TestHealthz(t *testing.T) {
	service := newTestService(t, func(http.ResponseWriter, *http.Request) {})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "ok\n" {
		t.Fatalf("health response = %d %q", response.Code, response.Body.String())
	}
}

func TestKleinGenerationConvertsToOpenAIResponse(t *testing.T) {
	service := newTestService(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/flux.2-klein-4b" {
			t.Fatalf("upstream path = %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization was not forwarded")
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload["width"] != float64(1024) || payload["height"] != float64(1024) || payload["steps"] != float64(4) {
			t.Fatalf("unexpected payload: %#v", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"artifacts":[{"base64":"`+tinyPNGBase64+`"}]}`)
	})

	request := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"flux.2-klein-4b","prompt":"a potato","response_format":"b64_json"}`))
	request.Header.Set("Authorization", "Bearer test-key")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if got := jsonPathString(t, response.Body.Bytes(), "b64_json"); got != tinyPNGBase64 {
		t.Fatalf("b64_json mismatch")
	}
}

func TestKontextGenerationUsesPlaceholderAndURLFormat(t *testing.T) {
	service := newTestService(t, func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload["image"] != kontextPlaceholder || payload["aspect_ratio"] != "1:1" {
			t.Fatalf("unexpected Kontext payload: %#v", payload)
		}
		_, _ = io.WriteString(w, `{"artifacts":[{"base64":"`+tinyPNGBase64+`"}]}`)
	})

	request := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"flux.1-kontext-dev","prompt":"a poster","size":"1024x1024","response_format":"url"}`))
	request.Header.Set("Authorization", "Bearer test-key")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if got := jsonPathString(t, response.Body.Bytes(), "url"); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Fatalf("unexpected url response: %q", got)
	}
}

func TestKontextMultipartEditRejectedWithoutUpstreamRetry(t *testing.T) {
	imageBytes, err := base64.StdEncoding.DecodeString(tinyPNGBase64)
	if err != nil {
		t.Fatal(err)
	}
	service := newTestService(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("hosted NVIDIA preview does not accept arbitrary edit images; upstream must not be called")
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	_ = writer.WriteField("model", "flux.1-kontext-dev")
	_ = writer.WriteField("prompt", "turn it blue")
	part, err := writer.CreateFormFile("image", "input.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(imageBytes)
	_ = writer.Close()

	request := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
	request.Header.Set("Authorization", "Bearer test-key")
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var payload openAIErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Error.Code != "image_edit_not_supported" {
		t.Fatalf("error code = %q", payload.Error.Code)
	}
}

func TestKleinMultipartEditRejectedWithoutUpstreamRetry(t *testing.T) {
	imageBytes, err := base64.StdEncoding.DecodeString(tinyPNGBase64)
	if err != nil {
		t.Fatal(err)
	}
	service := newTestService(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("hosted NVIDIA preview does not accept arbitrary edit images; upstream must not be called")
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	_ = writer.WriteField("model", "flux.2-klein-4b")
	_ = writer.WriteField("prompt", "turn it blue")
	part, err := writer.CreateFormFile("image", "input.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(imageBytes)
	_ = writer.Close()

	request := httptest.NewRequest(http.MethodPost, "/v1/images/edits", &body)
	request.Header.Set("Authorization", "Bearer test-key")
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var payload openAIErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Error.Code != "image_edit_not_supported" {
		t.Fatalf("error code = %q", payload.Error.Code)
	}
}

func TestUpstreamCredentialErrorIsSanitizedAndPreserved(t *testing.T) {
	service := newTestService(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"secret":"must-not-leak"}`)
	})
	request := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"flux.2-klein-4b","prompt":"a potato"}`))
	request.Header.Set("Authorization", "Bearer bad-key")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusPaymentRequired {
		t.Fatalf("status = %d", response.Code)
	}
	if strings.Contains(response.Body.String(), "must-not-leak") {
		t.Fatal("upstream body leaked")
	}
}

func TestRejectsMultipleImagesPerRequest(t *testing.T) {
	service := newTestService(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("upstream should not be called")
	})
	request := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"flux.2-klein-4b","prompt":"a potato","n":2}`))
	request.Header.Set("Authorization", "Bearer test-key")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	service.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", response.Code)
	}
}

func jsonPathString(t *testing.T, body []byte, key string) string {
	t.Helper()
	var payload struct {
		Data []map[string]string `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Data) != 1 {
		t.Fatalf("unexpected data: %#v", payload.Data)
	}
	return payload.Data[0][key]
}
