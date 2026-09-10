# url_is_absolute

Tests whether the text names a scheme. A URL with a scheme resolves without a base. A reference beginning with a slash or a path segment is relative and requires url_resolve before it names a resource.

## Syntax

```sql
url_is_absolute(text)
```

## Returns

BOOLEAN. NULL when the argument is NULL.

## Examples

```sql
SELECT url_is_absolute('/a/b')
```

false.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_resolve](url-resolve.md)
- [url_scheme](url-scheme.md)
