# url_resolve

Resolves a relative reference against a base URL, following the rules a browser applies. A reference beginning with a slash replaces the base's whole path, one beginning with a scheme replaces the base entirely, and one beginning with a dot segment is resolved relative to the base's directory.

## Syntax

```sql
url_resolve(base, relative)
```

## Returns

TEXT. NULL when either argument is NULL or the base is not a URL.

## Examples

```sql
SELECT url_resolve('https://example.com/a/b', '../c')
```

The text https://example.com/c.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_normalize](url-normalize.md)
- [url_is_absolute](url-is-absolute.md)
