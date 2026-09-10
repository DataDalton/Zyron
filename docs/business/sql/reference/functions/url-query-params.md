# url_query_params

Returns all parameters with their percent-encoding decoded, keeping repeated keys as separate pairs. Reading every parameter at once parses the URL once, where a call to url_query_param per key parses it per key.

## Syntax

```sql
url_query_params(url)
```

## Returns

A MAP of key to value. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_query_params('https://example.com/a?b=1&c=2')
```

Pairs for b and c.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_query_param](url-query-param.md)
- [url_parse](url-parse.md)
