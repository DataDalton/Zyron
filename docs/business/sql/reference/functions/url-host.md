# url_host

Returns the host in lower case, excluding any port. An IPv6 host is returned without the square brackets the URL syntax requires around it.

## Syntax

```sql
url_host(text)
```

## Returns

TEXT. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_host('https://Example.com:8443/a')
```

The text example.com.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_port](url-port.md)
- [url_domain](url-domain.md)
- [url_tld](url-tld.md)
