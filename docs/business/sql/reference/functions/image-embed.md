# image_embed

Returns a vector placing the image in an embedding space, so two images can be compared by vector distance rather than by pixels. Backed by an external model, so the call fails naming the tool and the configuration key when none is configured. The result suits a vector index.

## Syntax

```sql
image_embed(image [, model])
```

## Returns

VECTOR. NULL when the payload is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `model` | Model name to embed with. | clip. |

## Examples

```sql
SELECT image_embed(photo) FROM zyron_test.photos
```

One embedding vector per photo.

## Refused

- No embedding model is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cosine_similarity](cosine-similarity.md)
- [image_metadata](image-metadata.md)
