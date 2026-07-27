# OWASP Security Best Practices

## OWASP Top 10 (2021)

### A01 — Broken Access Control
Failures in enforcing user permissions. Users act outside their intended roles.
- **Prevention:** Implement role-based access control (RBAC). Verify permissions on every request. Deny by default.
- **In FastAPI:** Use `Depends()` with role-checking dependencies. Never rely on frontend-only restrictions.

### A02 — Cryptographic Failures
Exposure of sensitive data due to weak or missing encryption.
- **Prevention:** Use HTTPS everywhere. Encrypt sensitive data at rest. Use strong hashing (argon2) for passwords.
- **In FastAPI:** Use `HTTPSRedirectMiddleware`. Hash passwords with `argon2-cffi`. Never log passwords or tokens.

### A03 — Injection
Untrusted data sent to an interpreter (SQL, NoSQL, OS, LDAP).
- **Prevention:** Use parameterized queries. Validate and sanitize all input. Use an ORM.
- **In FastAPI:** SQLModel/SQLAlchemy parameterized queries. Pydantic validation rejects malformed input. Never use raw SQL with string formatting.

### A04 — Insecure Design
Missing or inadequate security controls in the design phase.
- **Prevention:** Threat modeling. Rate limiting. Secure defaults. Principle of least privilege.
- **In FastAPI:** Use `pyrate-limiter` on sensitive endpoints. Set secure cookie flags. Validate all inputs with Pydantic.

### A05 — Security Misconfiguration
Unnecessarily enabled features, default credentials, verbose error messages.
- **Prevention:** Disable debug mode in production. Custom error handlers. Remove default accounts.
- **In FastAPI:** Set `DEBUG=False` in production. Custom exception handlers that don't leak internals. Use `.env` for secrets.

### A06 — Vulnerable and Outdated Components
Using libraries with known vulnerabilities.
- **Prevention:** Regular updates. Dependency scanning. Minimum viable dependencies.
- **In FastAPI:** Pin versions in `requirements.txt`. Use `pip-audit` or Dependabot. Review dependencies regularly.

### A07 — Identification and Authentication Failures
Weak login mechanisms, session fixation, missing MFA.
- **Prevention:** Strong password policies. Rate-limit login attempts. Session invalidation on logout.
- **In FastAPI:** JWT with short expiration. Argon2 password hashing. Rate limiting on `/auth/login`.

### A08 — Software and Data Integrity Failures
Unverified updates, insecure CI/CD, unsigned artifacts.
- **Prevention:** Sign artifacts. Verify package checksums. Use trusted registries.
- **In Docker:** Pin base image digests. Verify pip package signatures. Use `--no-cache-dir`.

### A09 — Security Logging and Monitoring Failures
Insufficient logging, missing alerts, unmonitored suspicious activity.
- **Prevention:** Log all auth events. Monitor for anomalies. Structured logging.
- **In FastAPI:** Log authentication failures, rate limit hits, admin actions. Use structured logging.

### A10 — Server-Side Request Forgery (SSRF)
Server fetches a user-controlled URL, enabling access to internal systems.
- **Prevention:** Validate and whitelist URLs. Disable internal network access from application containers. Restrict redirects.
- **In FastAPI:** Never pass user input directly to HTTP clients without validation. Use URL allowlists for webhook features.

## General Security Guidelines

### Authentication
- Hash passwords with argon2 (`argon2-cffi`) — do not use MD5, SHA1, or unsalted SHA256
- JWT tokens should have short expiry (15-30 min) with refresh tokens
- Store tokens in `httpOnly` cookies, not `localStorage`
- Rate-limit login attempts (5 failures → 15 min block)

### Authorization
- Check permissions on every request, not just at login
- Use dependency injection for role checks in FastAPI
- Log all authorization failures
- Principle of least privilege for database users

### Input Validation
- Always validate with Pydantic schemas (types, ranges, formats)
- Reject unexpected fields (Pydantic `model_config = {"extra": "forbid"}`)
- Sanitize strings for XSS if rendering in HTML
- Limit string lengths and numeric ranges

### Database Security
- Use parameterized queries (SQLModel does this automatically)
- Database user should have minimal permissions (no `DROP TABLE`, no `CREATE` beyond migrations)
- Encrypt sensitive columns at application level
- Regular backups

### API Security
- Use HTTPS in production
- Rate limiting on all auth and admin endpoints
- CORS configured to specific origins, not `*`
- Security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`
- Version your API to manage breaking changes

### Container Security
- Don't run containers as root (use `USER` in Dockerfile)
- Read-only root filesystem where possible
- Scan images for vulnerabilities
- Use specific image tags, not `latest`
- Separate networks for different trust levels

## FastAPI-Specific Security Middleware

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
