# extract_urls

Scans the text and returns the URLs it recognises, in the order they appear. Trailing punctuation next to a URL is a common source of error, so check a result against url_parse before following it.

## Syntax

```sql
extract_urls(text)
```

## Returns

ARRAY of strings as JSON text. NULL when the text is NULL.

## Examples

```sql
SELECT extract_urls('see https://example.com/a and http://b.test')
```

Both URLs, in the order they appear.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [extract_emails](extract-emails.md)
- [url_parse](url-parse.md)
- [validate_url](validate-url.md)
