# CPA NVIDIA Image Middleware

Small OpenAI Images compatibility layer for NVIDIA-hosted FLUX endpoints.

Supported public models:

- `flux.2-klein-4b`: text-to-image through `POST /v1/images/generations`.
- `flux.1-kontext-dev`: text-to-image through `POST /v1/images/generations`.

The hosted NVIDIA Preview API does not accept arbitrary uploaded images for editing. It only accepts NVIDIA's predefined `example_id` images, so this compatibility service rejects `POST /v1/images/edits` with HTTP 400 instead of forwarding a request that would always fail and be retried by NewAPI. Arbitrary image editing requires a self-hosted NVIDIA NIM or another upstream.

The service accepts the NVIDIA credential forwarded by CPA, converts the request to NVIDIA's model-specific native schema, calls the native GenAI endpoint directly, and returns an OpenAI-compatible `data[].b64_json` or data URL response. It never logs prompts, images, credentials, request bodies, or upstream response bodies.

Production shape:

```text
NewAPI -> CPA openai-compatibility image provider -> this service -> NVIDIA
```

Default listen address: `172.17.0.1:18357`.

Set `UPSTREAM_PROXY_URL` when NVIDIA requests must use an outbound proxy. Keep proxy credentials in a root-only systemd environment file rather than in this repository.
