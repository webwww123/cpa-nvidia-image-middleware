package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	kleinModel           = "flux.2-klein-4b"
	kontextModel         = "flux.1-kontext-dev"
	defaultUpstreamBase  = "https://ai.api.nvidia.com/v1/genai/black-forest-labs"
	defaultListenAddress = "172.17.0.1:18357"
	maxRequestBodyBytes  = 40 << 20
	maxImageBytes        = 32 << 20
	kontextPlaceholder   = "data:image/png;example_id,0"
)

type serviceConfig struct {
	ListenAddress  string
	UpstreamBase   string
	RequestTimeout time.Duration
}

type imageService struct {
	upstreamBase        *url.URL
	client              *http.Client
	requestTimeout      time.Duration
	trustedNewAPIErrors bool
}

type imageRequest struct {
	Model          string   `json:"model"`
	Prompt         string   `json:"prompt"`
	N              *int     `json:"n,omitempty"`
	Size           string   `json:"size,omitempty"`
	ResponseFormat string   `json:"response_format,omitempty"`
	Width          *int     `json:"width,omitempty"`
	Height         *int     `json:"height,omitempty"`
	CFGScale       *float64 `json:"cfg_scale,omitempty"`
	Steps          *int     `json:"steps,omitempty"`
	Samples        *int     `json:"samples,omitempty"`
	Seed           *int64   `json:"seed,omitempty"`
	AspectRatio    string   `json:"aspect_ratio,omitempty"`
	Image          any      `json:"image,omitempty"`
}

type openAIErrorEnvelope struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func main() {
	listen := flag.String("listen", envOrDefault("LISTEN_ADDR", defaultListenAddress), "listen address")
	upstream := flag.String("upstream", envOrDefault("UPSTREAM_BASE_URL", defaultUpstreamBase), "NVIDIA image upstream base URL")
	proxy := flag.String("proxy", envOrDefault("UPSTREAM_PROXY_URL", ""), "optional proxy URL for NVIDIA upstream requests")
	timeout := flag.Duration("request-timeout", envDurationOrDefault("REQUEST_TIMEOUT", 75*time.Second), "maximum upstream request duration")
	trustedNewAPIErrors := flag.Bool("trusted-newapi-error-diagnostics", envBoolOrDefault("TRUSTED_NEWAPI_ERROR_DIAGNOSTICS", false), "return secret-scrubbed upstream HTTP diagnostics to a trusted NewAPI hop")
	flag.Parse()

	service, err := newImageServiceWithProxyAndDiagnostics(*upstream, *proxy, *timeout, nil, *trustedNewAPIErrors)
	if err != nil {
		log.Fatalf("configure service: %v", err)
	}

	server := &http.Server{
		Addr:              *listen,
		Handler:           service,
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	log.Printf("starting NVIDIA image middleware listen=%s upstream_host=%s", *listen, service.upstreamBase.Host)
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	stopCh := make(chan os.Signal, 1)
	signal.Notify(stopCh, syscall.SIGINT, syscall.SIGTERM)
	select {
	case sig := <-stopCh:
		log.Printf("stopping NVIDIA image middleware signal=%s", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("graceful shutdown failed: %v", err)
			_ = server.Close()
		}
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}
}

func newImageService(upstream string, timeout time.Duration, client *http.Client) (*imageService, error) {
	return newImageServiceWithProxy(upstream, "", timeout, client)
}

func newImageServiceWithProxy(upstream, proxy string, timeout time.Duration, client *http.Client) (*imageService, error) {
	return newImageServiceWithProxyAndDiagnostics(upstream, proxy, timeout, client, false)
}

func newImageServiceWithProxyAndDiagnostics(upstream, proxy string, timeout time.Duration, client *http.Client, trustedNewAPIErrors bool) (*imageService, error) {
	parsed, err := url.Parse(strings.TrimRight(strings.TrimSpace(upstream), "/"))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("invalid upstream base URL")
	}
	var proxyFunc func(*http.Request) (*url.URL, error)
	if strings.TrimSpace(proxy) != "" {
		parsedProxy, proxyErr := url.Parse(strings.TrimSpace(proxy))
		if proxyErr != nil || parsedProxy.Scheme == "" || parsedProxy.Host == "" {
			return nil, fmt.Errorf("invalid upstream proxy URL")
		}
		proxyFunc = http.ProxyURL(parsedProxy)
	}
	if timeout <= 0 {
		timeout = 75 * time.Second
	}
	if client == nil {
		client = &http.Client{Transport: &http.Transport{
			Proxy:                 proxyFunc,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   32,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: timeout,
		}}
	}
	return &imageService{upstreamBase: parsed, client: client, requestTimeout: timeout, trustedNewAPIErrors: trustedNewAPIErrors}, nil
}

func (s *imageService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
		return
	}

	started := time.Now()
	status := http.StatusNotFound
	upstreamStatus := 0
	model := ""
	defer func() {
		log.Printf("image_request method=%s path=%s model=%s status=%d upstream_status=%d duration_ms=%d", r.Method, r.URL.Path, model, status, upstreamStatus, time.Since(started).Milliseconds())
	}()

	if r.Method != http.MethodPost {
		status = http.StatusMethodNotAllowed
		writeError(w, status, "method not allowed", "method_not_allowed")
		return
	}

	path := strings.TrimRight(r.URL.Path, "/")
	if path != "/v1/images/generations" && path != "/images/generations" && path != "/v1/images/edits" && path != "/images/edits" {
		writeError(w, status, "not found", "not_found")
		return
	}

	authorization := strings.TrimSpace(r.Header.Get("Authorization"))
	if !strings.HasPrefix(strings.ToLower(authorization), "bearer ") || strings.TrimSpace(authorization[len("Bearer "):]) == "" {
		status = http.StatusUnauthorized
		writeError(w, status, "missing upstream credential", "unauthorized")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	request, err := decodeImageRequest(r, strings.HasSuffix(path, "/edits"))
	if err != nil {
		status = http.StatusBadRequest
		writeError(w, status, err.Error(), "invalid_request")
		return
	}
	model = canonicalModel(request.Model)
	request.Model = model
	if model == "" {
		status = http.StatusBadRequest
		writeError(w, status, "unsupported image model", "model_not_supported")
		return
	}
	if strings.HasSuffix(path, "/edits") {
		status = http.StatusBadRequest
		writeError(w, status, "image editing is unavailable on the NVIDIA hosted preview API; use image generation instead", "image_edit_not_supported")
		return
	}

	payload, err := buildNVIDIAPayload(request, strings.HasSuffix(path, "/edits"))
	if err != nil {
		status = http.StatusBadRequest
		writeError(w, status, err.Error(), "invalid_request")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.requestTimeout)
	defer cancel()
	upstreamBody, code, err := s.callUpstream(ctx, authorization, model, payload)
	upstreamStatus = code
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			status = http.StatusGatewayTimeout
			writeError(w, status, "image generation timed out", "upstream_timeout")
			return
		}
		status = http.StatusBadGateway
		writeError(w, status, "image generation upstream unavailable", "upstream_error")
		return
	}
	if code < http.StatusOK || code >= http.StatusMultipleChoices {
		status = publicUpstreamStatus(code)
		if s.trustedNewAPIErrors {
			writeTrustedUpstreamError(w, status, upstreamBody)
			return
		}
		message, errorCode := publicUpstreamError(status)
		writeError(w, status, message, errorCode)
		return
	}

	images, err := extractImagePayloads(upstreamBody)
	if err != nil || len(images) == 0 {
		status = http.StatusBadGateway
		writeError(w, status, "image generation returned no valid image", "invalid_upstream_response")
		return
	}

	response := map[string]any{"created": time.Now().Unix(), "data": make([]map[string]string, 0, len(images))}
	for _, image := range images {
		item := map[string]string{}
		if strings.EqualFold(strings.TrimSpace(request.ResponseFormat), "url") {
			item["url"] = "data:" + image.MIMEType + ";base64," + image.Base64
		} else {
			item["b64_json"] = image.Base64
		}
		response["data"] = append(response["data"].([]map[string]string), item)
	}

	status = http.StatusOK
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}

func decodeImageRequest(r *http.Request, edit bool) (imageRequest, error) {
	contentType := r.Header.Get("Content-Type")
	mediaType, _, _ := mime.ParseMediaType(contentType)
	if strings.HasPrefix(strings.ToLower(mediaType), "multipart/") {
		if !edit {
			return imageRequest{}, fmt.Errorf("multipart requests require the image edits endpoint")
		}
		return decodeMultipartImageRequest(r)
	}

	var request imageRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&request); err != nil {
		return imageRequest{}, fmt.Errorf("invalid image request JSON")
	}
	if strings.TrimSpace(request.Prompt) == "" {
		return imageRequest{}, fmt.Errorf("prompt is required")
	}
	if edit && imageString(request.Image) == "" {
		return imageRequest{}, fmt.Errorf("image is required for editing")
	}
	return request, nil
}

func decodeMultipartImageRequest(r *http.Request) (imageRequest, error) {
	if err := r.ParseMultipartForm(maxImageBytes); err != nil {
		return imageRequest{}, fmt.Errorf("invalid multipart image request")
	}
	if r.MultipartForm != nil {
		defer func() { _ = r.MultipartForm.RemoveAll() }()
	}
	request := imageRequest{
		Model:          strings.TrimSpace(r.FormValue("model")),
		Prompt:         strings.TrimSpace(r.FormValue("prompt")),
		Size:           strings.TrimSpace(r.FormValue("size")),
		ResponseFormat: strings.TrimSpace(r.FormValue("response_format")),
	}
	if request.Prompt == "" {
		return imageRequest{}, fmt.Errorf("prompt is required")
	}
	fileHeader, err := firstImageFile(r)
	if err != nil {
		return imageRequest{}, err
	}
	file, err := fileHeader.Open()
	if err != nil {
		return imageRequest{}, fmt.Errorf("open image upload failed")
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, maxImageBytes+1))
	if err != nil || len(raw) == 0 || len(raw) > maxImageBytes {
		return imageRequest{}, fmt.Errorf("invalid image upload")
	}
	mimeType := strings.TrimSpace(fileHeader.Header.Get("Content-Type"))
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		mimeType = http.DetectContentType(raw)
	}
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		return imageRequest{}, fmt.Errorf("uploaded file is not an image")
	}
	request.Image = "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(raw)
	return request, nil
}

func firstImageFile(r *http.Request) (*multipart.FileHeader, error) {
	if r.MultipartForm == nil {
		return nil, fmt.Errorf("image is required for editing")
	}
	for _, key := range []string{"image", "image[]"} {
		if files := r.MultipartForm.File[key]; len(files) > 0 && files[0] != nil {
			return files[0], nil
		}
	}
	return nil, fmt.Errorf("image is required for editing")
}

func canonicalModel(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if slash := strings.LastIndex(model, "/"); slash >= 0 {
		model = strings.TrimSpace(model[slash+1:])
	}
	switch model {
	case kleinModel, kontextModel:
		return model
	default:
		return ""
	}
}

func buildNVIDIAPayload(request imageRequest, edit bool) (map[string]any, error) {
	if strings.TrimSpace(request.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if request.N != nil && *request.N != 1 {
		return nil, fmt.Errorf("n must be 1; use concurrent requests for multiple images")
	}
	if request.Samples != nil && *request.Samples != 1 {
		return nil, fmt.Errorf("samples must be 1")
	}
	seed := randomSeed()
	if request.Seed != nil {
		seed = *request.Seed
	}

	switch request.Model {
	case kleinModel:
		width, height, err := imageDimensions(request)
		if err != nil {
			return nil, err
		}
		cfgScale := 1.0
		if request.CFGScale != nil {
			cfgScale = *request.CFGScale
		}
		if cfgScale < 1 || cfgScale > 20 {
			return nil, fmt.Errorf("cfg_scale must be between 1 and 20")
		}
		steps := 4
		if request.Steps != nil {
			steps = *request.Steps
		}
		if steps < 1 || steps > 50 {
			return nil, fmt.Errorf("steps must be between 1 and 50")
		}
		return map[string]any{
			"prompt":    request.Prompt,
			"height":    height,
			"width":     width,
			"cfg_scale": cfgScale,
			"samples":   1,
			"seed":      seed,
			"steps":     steps,
			"image":     nil,
		}, nil
	case kontextModel:
		cfgScale := 3.5
		if request.CFGScale != nil {
			cfgScale = *request.CFGScale
		}
		if cfgScale < 1 || cfgScale > 20 {
			return nil, fmt.Errorf("cfg_scale must be between 1 and 20")
		}
		steps := 30
		if request.Steps != nil {
			steps = *request.Steps
		}
		if steps < 1 || steps > 50 {
			return nil, fmt.Errorf("steps must be between 1 and 50")
		}
		image := imageString(request.Image)
		if image == "" {
			if edit {
				return nil, fmt.Errorf("image is required for editing")
			}
			image = kontextPlaceholder
		}
		aspectRatio := strings.TrimSpace(request.AspectRatio)
		if aspectRatio == "" {
			aspectRatio = aspectRatioFromSize(request.Size)
		}
		if aspectRatio == "" {
			aspectRatio = "match_input_image"
		}
		return map[string]any{
			"prompt":       request.Prompt,
			"cfg_scale":    cfgScale,
			"aspect_ratio": aspectRatio,
			"samples":      1,
			"seed":         seed,
			"steps":        steps,
			"image":        image,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported image model")
	}
}

func imageDimensions(request imageRequest) (int, int, error) {
	width, height := 1024, 1024
	if strings.TrimSpace(request.Size) != "" {
		parts := strings.Split(strings.ToLower(strings.TrimSpace(request.Size)), "x")
		if len(parts) != 2 {
			return 0, 0, fmt.Errorf("size must use WIDTHxHEIGHT")
		}
		parsedWidth, errWidth := strconv.Atoi(parts[0])
		parsedHeight, errHeight := strconv.Atoi(parts[1])
		if errWidth != nil || errHeight != nil {
			return 0, 0, fmt.Errorf("size must use WIDTHxHEIGHT")
		}
		width, height = parsedWidth, parsedHeight
	}
	if request.Width != nil {
		width = *request.Width
	}
	if request.Height != nil {
		height = *request.Height
	}
	if width < 256 || width > 2048 || height < 256 || height > 2048 || width%64 != 0 || height%64 != 0 {
		return 0, 0, fmt.Errorf("width and height must be multiples of 64 between 256 and 2048")
	}
	return width, height, nil
}

func aspectRatioFromSize(size string) string {
	switch strings.ToLower(strings.TrimSpace(size)) {
	case "1024x1024", "1:1":
		return "1:1"
	case "1792x1024", "16:9":
		return "16:9"
	case "1024x1792", "9:16":
		return "9:16"
	case "1536x1024", "3:2":
		return "3:2"
	case "1024x1536", "2:3":
		return "2:3"
	default:
		return ""
	}
}

func imageString(value any) string {
	text, _ := value.(string)
	return strings.TrimSpace(text)
}

func randomSeed() int64 {
	var buffer [8]byte
	if _, err := rand.Read(buffer[:]); err == nil {
		return int64(binary.LittleEndian.Uint64(buffer[:]) & 0x7fffffff)
	}
	return time.Now().UnixNano() & 0x7fffffff
}

func (s *imageService) callUpstream(ctx context.Context, authorization, model string, payload map[string]any) ([]byte, int, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, err
	}
	endpoint := *s.upstreamBase
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + "/" + model
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(encoded))
	if err != nil {
		return nil, 0, err
	}
	request.Header.Set("Authorization", authorization)
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", "cpa-nvidia-image-middleware")
	response, err := s.client.Do(request)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxRequestBodyBytes+1))
	if err != nil || len(body) > maxRequestBodyBytes {
		return nil, response.StatusCode, fmt.Errorf("read upstream response failed")
	}
	return body, response.StatusCode, nil
}

type extractedImage struct {
	Base64   string
	MIMEType string
}

func extractImagePayloads(body []byte) ([]extractedImage, error) {
	var payload any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	values := make([]string, 0, 4)
	collectImageStrings(payload, &values)
	images := make([]extractedImage, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		mimeType := ""
		if strings.HasPrefix(strings.ToLower(value), "data:") {
			prefix, encoded, ok := strings.Cut(value, ",")
			if !ok {
				continue
			}
			value = encoded
			mimeType = strings.TrimPrefix(strings.Split(prefix, ";")[0], "data:")
		}
		raw, err := base64.StdEncoding.DecodeString(value)
		if err != nil || len(raw) == 0 {
			continue
		}
		if mimeType == "" {
			mimeType = http.DetectContentType(raw)
		}
		if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		images = append(images, extractedImage{Base64: value, MIMEType: mimeType})
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("no image payload found")
	}
	return images, nil
}

func collectImageStrings(value any, output *[]string) {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"base64", "b64_json", "image"} {
			if text, ok := typed[key].(string); ok {
				*output = append(*output, text)
			}
		}
		for _, key := range []string{"artifacts", "data", "images", "output"} {
			if child, ok := typed[key]; ok {
				collectImageStrings(child, output)
			}
		}
	case []any:
		for _, child := range typed {
			collectImageStrings(child, output)
		}
	}
}

func publicUpstreamStatus(status int) int {
	if status >= 400 && status <= 599 {
		return status
	}
	return http.StatusBadGateway
}

func publicUpstreamError(status int) (string, string) {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return "upstream credential rejected", "upstream_unauthorized"
	case http.StatusPaymentRequired:
		return "upstream subscription unavailable", "upstream_payment_required"
	case http.StatusTooManyRequests:
		return "upstream rate limited", "upstream_rate_limited"
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return "invalid image request", "invalid_request"
	default:
		if status >= 500 {
			return "image generation upstream unavailable", "upstream_error"
		}
		return "image generation failed", "upstream_error"
	}
}

func writeError(w http.ResponseWriter, status int, message, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(openAIErrorEnvelope{Error: openAIError{Message: message, Type: "invalid_request_error", Code: code}})
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func envDurationOrDefault(name string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func envBoolOrDefault(name string, fallback bool) bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if value == "" {
		return fallback
	}
	switch value {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	default:
		return fallback
	}
}
