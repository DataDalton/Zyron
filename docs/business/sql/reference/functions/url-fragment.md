# url_fragment

Returns the part after the hash, excluding the hash itself. A URL with no fragment returns NULL. A fragment is never sent to a server, so it is present only in URLs recorded on the client side.

## Syntax

```sql
url_fragment(text)
```

## Returns

TEXT. NULL when no fragment is written.

## Examples

```sql
SELECT url_fragment('https://example.com/a#section')
```

The text section.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [url_path](url-path.md)
- [url_parse](url-parse.md)
