# url_tld

Returns the public suffix, which is the part of the host nobody can register a name directly beneath. For example.co.uk this is co.uk rather than uk, because uk is not registrable directly.

## Syntax

```sql
url_tld(text)
```

## Returns

TEXT. NULL when the host has no known suffix.

## Examples

```sql
SELECT url_tld('https://example.co.uk/x')
```

The text co.uk.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_domain](url-domain.md)
- [url_host](url-host.md)
