# url_scheme

Returns the scheme in lower case, with no trailing colon or slashes. Text that is not a URL returns NULL rather than raising an error.

## Syntax

```sql
url_scheme(text)
```

## Returns

TEXT. NULL when the argument is NULL or is not a URL.

## Examples

```sql
SELECT url_scheme('HTTPS://example.com/a')
```

The text https.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_parse](url-parse.md)
- [url_is_absolute](url-is-absolute.md)
