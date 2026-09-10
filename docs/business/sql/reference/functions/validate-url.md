# validate_url

Answers whether url_parse accepts the text. It checks structure rather than reachability, so a URL whose host does not resolve passes.

## Syntax

```sql
validate_url(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_url('https://example.com/a')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_parse](url-parse.md)
- [extract_urls](extract-urls.md)
- [url_is_absolute](url-is-absolute.md)
