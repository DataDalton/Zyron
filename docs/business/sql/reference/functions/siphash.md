# siphash

Computes SipHash, which is designed so an attacker who does not hold the key cannot construct colliding inputs. That makes it the hash to use where keys come from outside and a flood of collisions would degrade a hash table into a list. It costs more than xxhash64.

## Syntax

```sql
siphash(bytes)
```

## Returns

BIGINT. NULL when the argument is NULL.

## Examples

```sql
SELECT siphash('hello'::BYTEA)
```

A BIGINT hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xxhash64](xxhash64.md)
- [cityhash64](cityhash64.md)
