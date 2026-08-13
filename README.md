# CPA NVIDIA Image Middleware

Small OpenAI Images compatibility layer for NVIDIA-hosted FLUX endpoints.

Supported public models:

- `flux.2-klein-4b`: text-to-image through `POST /v1/images/generations`.
- `flux.1-kontext-dev`: text-to-image and image editing through `POST /v1/images/generations` and `POST /v1/images/edits`.

The service accepts the NVIDIA credential forwarded by CPA, converts the request to NVIDIA's model-specific native schema, calls the native GenAI endpoint directly, and returns an OpenAI-compatible `data[].b64_json` or data URL response. It never logs prompts, images, credentials, request bodies, or upstream response bodies.

Production shape:

```text
NewAPI -> CPA openai-compatibility image provider -> this service -> NVIDIA
```

Default listen address: `172.17.0.1:18357`.
