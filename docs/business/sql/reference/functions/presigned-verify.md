# presigned_verify

Checks the signature against the node's presign secret and the expiry against the current time. A bad signature and a passed expiry both answer false rather than raising an error, because the question asked is whether the link is good.

## Syntax

```sql
presigned_verify(url)
```

## Returns

BOOLEAN. NULL when the link is NULL.

## Examples

```sql
SELECT presigned_verify('https://example.invalid/media/abc?sig=wrong')
```

false.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [presigned_url](presigned-url.md)
