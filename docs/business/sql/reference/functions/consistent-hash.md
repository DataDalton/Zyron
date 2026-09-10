# consistent_hash

Maps the key to a bucket by jump consistent hashing, so raising the bucket count from n to n plus one moves only about one key in n plus one, where a modulo would move almost every key. Buckets are numbered from 0. A bucket count of 0 gives 0.

## Syntax

```sql
consistent_hash(key, num_buckets)
```

## Returns

INTEGER below the bucket count. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `num_buckets` | How many buckets to spread across. | Not applicable. |

## Examples

```sql
SELECT consistent_hash('user-42', 8) < 8
```

true, because the bucket is below the count.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hash_combine](hash-combine.md)
- [murmur3_32](murmur3-32.md)
- [fnv1a_64](fnv1a-64.md)
