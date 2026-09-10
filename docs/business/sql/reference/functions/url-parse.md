# url_parse

Parses a URL and returns its scheme, host, port, path, query and fragment together. Use it where several parts are needed, rather than calling url_scheme, url_host and url_path separately and parsing the same text three times.

## Syntax

```sql
url_parse(text)
```

## Returns

A STRUCT of the URL's parts. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_parse('https://example.com/a?b=1#c')
```

A value holding the scheme, host, path, query and fragment.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_scheme](url-scheme.md)
- [url_host](url-host.md)
- [url_path](url-path.md)
- [url_query_params](url-query-params.md)
