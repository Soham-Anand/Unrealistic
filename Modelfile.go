# Unrealistic v1 — Ollama-NATIVE Go template (proven: template presence is the
# variable; Jinja originals get mangled by Ollama's converter).
#   ollama create unrealistic-go -f Modelfile.go
FROM /Users/sohamanand/Unrealistic/gguf/unrealistic-v1-Q4_K_M.gguf
PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.25
PARAMETER num_ctx 1024
PARAMETER stop "User:"
TEMPLATE """{{- range .Messages }}{{ if eq .Role "user" }}User: {{ .Content }}
Assistant:{{ else }}{{ .Content }}</s>{{ end }}{{ end }}{{- if .Prompt }}User: {{ .Prompt }}
Assistant:{{ end }}"""
