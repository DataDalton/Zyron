# url_port

Returns the port the URL states. A URL relying on its scheme's default port returns NULL rather than that default, so an explicit port is distinguishable from an implied one.

## Syntax

```sql
url_port(text)
```

## Returns

TEXT. NULL when no port is written.

## Examples

```sql
SELECT url_port('https://example.com:8443/a')
```

The text 8443.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_host](url-host.md)
- [url_scheme](url-scheme.md)
