# qr_encode

Returns a PNG at error correction level M, which recovers from about 15 percent of the code being damaged. The level is fixed rather than a parameter. Text too long for the largest code gives an empty value rather than an error.

## Syntax

```sql
qr_encode(text)
```

## Returns

BYTEA holding a PNG, empty when the text cannot be encoded. NULL when the text is NULL.

## Examples

```sql
SELECT qr_decode(qr_encode('hello'))
```

hello.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [qr_decode](qr-decode.md)
- [barcode_encode](barcode-encode.md)
- [data_matrix_encode](data-matrix-encode.md)
