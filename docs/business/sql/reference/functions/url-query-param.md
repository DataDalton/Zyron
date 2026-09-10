# url_query_param

Returns the value of the named parameter with its percent-encoding decoded. Where a key appears several times the first value is returned. A key with no value returns the empty string, which is distinguishable from a key that is absent and returns NULL.

## Syntax

```sql
url_query_param(url, key)
```

## Returns

TEXT. NULL when the key is absent.

## Examples

```sql
SELECT url_query_param('https://example.com/a?b=1&c=2', 'c')
```

The text 2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_query_params](url-query-params.md)
- [url_path](url-path.md)
