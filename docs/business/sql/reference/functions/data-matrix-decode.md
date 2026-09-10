# data_matrix_decode

Reads the code in the image and returns its text. An image holding no readable code gives an empty string rather than NULL.

## Syntax

```sql
data_matrix_decode(image)
```

## Returns

VARCHAR, empty when no code is read. NULL when the image is NULL.

## Examples

```sql
SELECT data_matrix_decode(data_matrix_encode('PART-1'))
```

PART-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [data_matrix_encode](data-matrix-encode.md)
- [qr_decode](qr-decode.md)
