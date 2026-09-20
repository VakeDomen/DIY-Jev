# Security and deployment

This is an experimental local inference server, not a hardened public service.
The default listener is loopback. There is no built-in authentication, TLS,
per-client rate limiting, or tenant isolation. Use an authenticated reverse proxy
with request limits and timeouts before allowing remote access.

Treat models and benchmark dependencies as trusted executable supply-chain inputs.
Do not commit access tokens, private request data, or local environment files.
Before publishing, scan the complete Git history with a secret scanner and rotate
any exposed credentials. A working-tree scan does not establish a clean history.

For vulnerabilities, use the hosting platform's private reporting facility if
enabled. Do not post credentials or private user data in public issues.
