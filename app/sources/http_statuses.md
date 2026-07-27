# HTTP Status Codes

## 1xx — Informational

| Code | Name | Description |
|------|------|-------------|
| 100 | Continue | Client should continue with the request |
| 101 | Switching Protocols | Server is switching protocols as requested |
| 102 | Processing | Server has received and is processing the request (WebDAV) |
| 103 | Early Hints | Used with Link header to allow preloading before final response |

## 2xx — Successful

| Code | Name | Description |
|------|------|-------------|
| 200 | OK | Standard success response |
| 201 | Created | Resource was successfully created (POST) |
| 202 | Accepted | Request accepted but processing not complete |
| 203 | Non-Authoritative Information | Returning modified metadata from a third-party source |
| 204 | No Content | Success but no body to return (DELETE) |
| 205 | Reset Content | Tells the user agent to reset the document view |
| 206 | Partial Content | Partial GET request successful (range headers) |
| 207 | Multi-Status | Multiple status codes for multiple operations (WebDAV) |
| 208 | Already Reported | Members of a DAV binding were already enumerated |
| 226 | IM Used | Server fulfilled request using instance manipulations |

## 3xx — Redirection

| Code | Name | Description |
|------|------|-------------|
| 300 | Multiple Choices | Multiple representations available |
| 301 | Moved Permanently | Resource has a new permanent URL |
| 302 | Found | Resource temporarily at a different URL |
| 303 | See Other | Response available at another URL (GET) |
| 304 | Not Modified | Cached version is still valid |
| 305 | Use Proxy | Must use a proxy (deprecated) |
| 307 | Temporary Redirect | Temporary redirect, method unchanged |
| 308 | Permanent Redirect | Permanent redirect, method unchanged |

## 4xx — Client Error

| Code | Name | Description |
|------|------|-------------|
| 400 | Bad Request | Malformed request syntax or invalid parameters |
| 401 | Unauthorized | Authentication required or failed |
| 402 | Payment Required | Reserved for future use |
| 403 | Forbidden | Server understands but refuses to authorize |
| 404 | Not Found | Resource does not exist |
| 405 | Method Not Allowed | HTTP method not supported for this resource |
| 406 | Not Acceptable | Cannot produce response matching Accept headers |
| 407 | Proxy Authentication Required | Must authenticate with proxy |
| 408 | Request Timeout | Server timed out waiting for request |
| 409 | Conflict | Request conflicts with current state |
| 410 | Gone | Resource permanently deleted |
| 411 | Length Required | Content-Length header required |
| 412 | Precondition Failed | Precondition headers evaluated false |
| 413 | Payload Too Large | Request body exceeds server limits |
| 414 | URI Too Long | Request URI exceeds server limits |
| 415 | Unsupported Media Type | Media type not supported |
| 416 | Range Not Satisfiable | Range header cannot be satisfied |
| 417 | Expectation Failed | Expect header cannot be satisfied |
| 418 | I'm a Teapot | HTCPCP teapot (RFC 2324, April Fools) |
| 422 | Unprocessable Entity | Request body is semantically invalid (FastAPI validation errors) |
| 423 | Locked | Resource is locked (WebDAV) |
| 424 | Failed Dependency | Request failed due to failure of a previous request (WebDAV) |
| 425 | Too Early | Server is unwilling to risk processing a replay attack |
| 426 | Upgrade Required | Client should upgrade to a different protocol |
| 428 | Precondition Required | Server requires conditional requests |
| 429 | Too Many Requests | Rate limit exceeded |
| 431 | Request Header Fields Too Large | Headers exceed server limits |
| 451 | Unavailable For Legal Reasons | Resource blocked for legal reasons |

## 5xx — Server Error

| Code | Name | Description |
|------|------|-------------|
| 500 | Internal Server Error | Generic server error |
| 501 | Not Implemented | Server does not support the functionality |
| 502 | Bad Gateway | Invalid response from upstream server |
| 503 | Service Unavailable | Server temporarily overloaded or under maintenance |
| 504 | Gateway Timeout | Upstream server did not respond in time |
| 505 | HTTP Version Not Supported | Server does not support the HTTP protocol version |
| 506 | Variant Also Negotiates | Server configuration error in transparent negotiation |
| 507 | Insufficient Storage | Server cannot store the representation (WebDAV) |
| 508 | Loop Detected | Infinite loop detected while processing (WebDAV) |
| 510 | Not Extended | Further extensions to the request are needed |
| 511 | Network Authentication Required | Client must authenticate to access the network |

## FastAPI-Specific Response Patterns

| Pattern | Code | Usage |
|---------|------|-------|
| Default success | 200 | `@app.get()` returns 200 by default |
| Created resource | 201 | Set `status_code=201` on POST endpoints |
| No content | 204 | Set `status_code=204` on DELETE endpoints |
| Redirect | 307/308 | Use `RedirectResponse` or `Response(status_code=307)` |
| Client error | 422 | Automatic — FastAPI validation errors return 422 |
| Auth failure | 401 | Raise `HTTPException(401)` or use `Depends()` with auth |
| Forbidden | 403 | Raise `HTTPException(403)` for insufficient permissions |
| Not found | 404 | Raise `HTTPException(404)` when resource missing |
| Rate limited | 429 | Raised by `pyrate-limiter` or custom middleware |
| Server error | 500 | Unhandled exceptions return 500 automatically |

## Common Response Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | Media type of the response body (`application/json`, `text/html`) |
| `Content-Length` | Size of the response body in bytes |
| `Location` | URL for redirects (3xx) or newly created resources (201) |
| `Set-Cookie` | Set cookies from the server |
| `WWW-Authenticate` | Challenge information for 401 responses |
| `Cache-Control` | Caching directives |
| `X-Request-ID` | Trace identifier for request (useful for debugging) |
| `Access-Control-Allow-Origin` | CORS header indicating allowed origins |
