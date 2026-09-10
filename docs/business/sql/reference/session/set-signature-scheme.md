# SET SIGNATURE SCHEME

Binds a signing algorithm to one kind of artifact, such as a token or a release. Each kind is bound separately, so one kind can move to a new algorithm while others stay. The binding is durable and survives a restart.

## Syntax

```sql
SET SIGNATURE SCHEME name FOR ARTIFACT KIND kind
```

## Examples

```sql
SET SIGNATURE SCHEME Ed25519 FOR ARTIFACT KIND JWT
```

Tokens are signed with that algorithm from now on.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ROTATE SIGNATURE SCHEME](rotate-signature-scheme.md)
- [ROTATE SERVICE PRINCIPAL KEY](rotate-service-principal-key.md)
