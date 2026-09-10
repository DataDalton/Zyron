# url_domain

Returns the domain a registrant holds, dropping subdomains. For a.b.example.co.uk the result is example.co.uk, which requires knowing that co.uk is a public suffix rather than a domain. Grouping by host counts each subdomain separately, and grouping by domain counts each site once.

## Syntax

```sql
url_domain(text)
```

## Returns

TEXT. NULL when the argument is NULL or holds no registrable domain.

## Examples

```sql
SELECT url_domain('https://a.b.example.com/x')
```

The text example.com.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_host](url-host.md)
- [url_tld](url-tld.md)
