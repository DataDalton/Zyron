# hmac_sha256

Returns the HMAC of the data under the key. Unlike a bare digest this cannot be recomputed without the key, so it proves the data came from a holder of the key rather than only that the data is unchanged.

## Syntax

```sql
hmac_sha256(bytes, key)
```

## Returns

BYTEA of 32 bytes. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `key` | Secret key. A key longer than the block size is hashed first, a shorter one is padded. | Not applicable. |

## Examples

```sql
SELECT length(hmac_sha256('message', 'secret'))
```

32.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sha256](sha256.md)
- [presigned_url](presigned-url.md)
