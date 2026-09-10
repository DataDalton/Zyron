# pca

Centres each column on its mean, then returns the requested number of components, the samples projected onto them, and the fraction of variance each accounts for. The input rows are samples and the columns are features, so a matrix built the other way round returns components of the wrong thing without failing. The columns are not scaled to equal variance, so a feature measured in larger units dominates the result.

## Syntax

```sql
pca(matrix, components)
```

## Returns

COMPOSITE as JSON text with the keys components, scores and variance_explained. NULL when the matrix or the count is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `components` | How many components to keep, between 1 and the feature count. | Not applicable. |

## Examples

```sql
SELECT pca(matrix_create(3, 2, '[1,2,3,6,5,10]'), 1)
```

An object holding components, scores and variance_explained, where variance_explained is [1.0] because the two features are proportional.

## Refused

- The component count is zero or above the feature count.
- The component count is negative or above the 32-bit range.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [svd](svd.md)
- [eigenvalues](eigenvalues.md)
- [matrix_create](matrix-create.md)
