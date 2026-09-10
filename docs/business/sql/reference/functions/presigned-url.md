# presigned_url

Addresses the payload by the SHA-256 of its bytes and signs that address, the method and an expiry with the node's presign secret. The link carries no grant of its own, so anyone holding it can use it until it expires, and the same payload always produces the same address.

## Syntax

```sql
presigned_url(payload, expires_in [, method])
```

## Returns

TEXT holding the signed link. NULL when the payload or the lifetime is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `expires_in` | Lifetime as a count and a unit suffix, where the suffixes are ms, s, m and h. | Not applicable. |
| `method` | HTTP method the link is signed for. | GET. |

## Examples

```sql
SELECT presigned_verify(presigned_url(photo, '15m')) FROM zyron_test.photos
```

true, because a link just issued has not expired.

## Refused

- The lifetime carries no unit suffix.
- The lifetime count is not a whole number.
- The lifetime is zero or negative.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [presigned_verify](presigned-verify.md)
- [image_metadata](image-metadata.md)
