# url_path

Returns the path with its leading slash, excluding the query string and the fragment. A URL with no path returns a single slash.

## Syntax

```sql
url_path(text)
```

## Returns

TEXT. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_path('https://example.com/a/b?c=1')
```

The text /a/b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_query_param](url-query-param.md)
- [url_fragment](url-fragment.md)
