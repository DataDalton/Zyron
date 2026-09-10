# url_normalize

Lower-cases the scheme and host, resolves dot segments in the path, removes a port matching the scheme's default, and orders the query parameters. Two URLs addressing the same resource normalize to the same text, so they group and deduplicate together.

## Syntax

```sql
url_normalize(text)
```

## Returns

TEXT. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_normalize('HTTP://Example.com:80/a/../b')
```

The text http://example.com/b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_resolve](url-resolve.md)
- [url_parse](url-parse.md)
