package main

import (
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

const maxTrustedErrorBytes = 64 << 10

var (
	trustedPrivateKeyPattern  = regexp.MustCompile(`(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----`)
	trustedBearerPattern      = regexp.MustCompile(`(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+`)
	trustedBasicPattern       = regexp.MustCompile(`(?i)\bBasic\s+[A-Za-z0-9+/]{4,}={0,2}`)
	trustedSKPattern          = regexp.MustCompile(`(?i)\bsk-[A-Za-z0-9_-]{8,}`)
	trustedJSONSecretPattern  = regexp.MustCompile(`(?i)("(?:authorization|proxy[_-]?authorization|auth|api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|private[_-]?key|password|passwd|pwd|passphrase|secret|credentials?|cookie|set[_-]?cookie|session[_-]?token)"\s*:\s*)"(?:\\.|[^"\\])*"`)
	trustedSecretPattern      = regexp.MustCompile(`(?i)\b(authorization|proxy[_-]?authorization|auth|api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|private[_-]?key|password|passwd|pwd|passphrase|secret|credentials?|cookie|set[_-]?cookie|session[_-]?token)\s*[:=]\s*[^\s,;&]+`)
	trustedQuerySecretPattern = regexp.MustCompile(`(?i)([?&](?:api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|private[_-]?key|password|passwd|pwd|passphrase|secret|auth|authorization|credentials?|session[_-]?token)=)[^&#\s]+`)
	trustedURLPattern         = regexp.MustCompile(`(?i)\bhttps?://[^\s"'<>]+`)
)

func writeTrustedUpstreamError(w http.ResponseWriter, status int, upstreamBody []byte) {
	body := trustedUpstreamErrorPayload(upstreamBody)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func trustedUpstreamErrorPayload(upstreamBody []byte) []byte {
	if len(upstreamBody) > maxTrustedErrorBytes {
		upstreamBody = upstreamBody[:maxTrustedErrorBytes]
	}
	var payload any
	if json.Unmarshal(upstreamBody, &payload) == nil {
		payload = scrubTrustedValue(payload)
		if object, ok := payload.(map[string]any); ok {
			if _, hasError := object["error"]; hasError {
				if body, err := json.Marshal(object); err == nil && len(body) <= maxTrustedErrorBytes {
					return body
				}
			}
			message := trustedErrorMessage(object)
			if message == "" {
				message = "upstream image request failed"
			}
			body, _ := json.Marshal(map[string]any{
				"error": map[string]any{
					"message":  message,
					"type":     "upstream_error",
					"code":     "upstream_error",
					"metadata": object,
				},
			})
			if len(body) <= maxTrustedErrorBytes {
				return body
			}
		}
	}
	message := scrubTrustedText(string(upstreamBody))
	if message == "" {
		message = "upstream image request failed"
	}
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "upstream_error",
			"code":    "upstream_error",
		},
	})
	return body
}

func trustedErrorMessage(payload map[string]any) string {
	for _, key := range []string{"message", "detail", "title", "error_description"} {
		if message, ok := payload[key].(string); ok && strings.TrimSpace(message) != "" {
			return message
		}
	}
	return ""
}

func scrubTrustedValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if sensitiveTrustedField(key) {
				typed[key] = "<redacted>"
				continue
			}
			typed[key] = scrubTrustedValue(child)
		}
		return typed
	case []any:
		for index, child := range typed {
			typed[index] = scrubTrustedValue(child)
		}
		return typed
	case string:
		return scrubTrustedText(typed)
	default:
		return value
	}
}

func scrubTrustedText(value string) string {
	value = strings.ToValidUTF8(value, "�")
	value = trustedPrivateKeyPattern.ReplaceAllString(value, "<redacted-private-key>")
	value = redactTrustedURLSecrets(value)
	value = trustedBearerPattern.ReplaceAllString(value, "Bearer <redacted>")
	value = trustedBasicPattern.ReplaceAllString(value, "Basic <redacted>")
	value = trustedSKPattern.ReplaceAllString(value, "sk-<redacted>")
	value = trustedJSONSecretPattern.ReplaceAllString(value, `$1"<redacted>"`)
	value = trustedSecretPattern.ReplaceAllString(value, "$1=<redacted>")
	value = trustedQuerySecretPattern.ReplaceAllString(value, "$1<redacted>")
	value = strings.TrimSpace(value)
	if len(value) > maxTrustedErrorBytes {
		value = value[:maxTrustedErrorBytes]
	}
	return value
}

func redactTrustedURLSecrets(value string) string {
	return trustedURLPattern.ReplaceAllStringFunc(value, func(raw string) string {
		parsed, err := url.Parse(raw)
		if err != nil || parsed.Scheme == "" || parsed.Host == "" {
			return "<redacted-url>"
		}
		parsed.User = nil
		query := parsed.Query()
		for key := range query {
			if sensitiveTrustedField(key) {
				query.Set(key, "<redacted>")
			}
		}
		parsed.RawQuery = query.Encode()
		parsed.Fragment = ""
		return parsed.String()
	})
}

func sensitiveTrustedField(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	switch value {
	case "authorization", "proxy_authorization", "auth", "api_key", "key", "token",
		"access_token", "refresh_token", "id_token", "client_secret", "private_key",
		"password", "passwd", "pwd", "passphrase", "secret", "credential", "credentials",
		"cookie", "set_cookie", "session_token":
		return true
	}
	for _, suffix := range []string{"_token", "_secret", "_password", "_credential", "_credentials", "_private_key"} {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}
