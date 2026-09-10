# detect_encoding

Reads any byte order mark and otherwise judges the encoding from the byte patterns, naming one of ascii, utf-8, utf-16le, utf-16be, utf-32le and utf-32be in lower case. Bytes that are all below 128 are reported as ascii rather than as utf-8, even though the two agree on that range. The answer is a best reading rather than a certainty, because several single-byte encodings are indistinguishable from their bytes alone.

## Syntax

```sql
detect_encoding(bytes)
```

## Returns

VARCHAR naming an encoding. NULL when the payload is NULL.

## Examples

```sql
SELECT detect_encoding('plain text')
```

ascii.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [detect_mime_type](detect-mime-type.md)
- [is_binary](is-binary.md)
