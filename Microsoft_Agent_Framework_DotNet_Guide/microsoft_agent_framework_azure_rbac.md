---
layout: default
title: Microsoft Agent Framework - Azure RBAC
description: Least-privileged RBAC for Azure AI Agents and Key Vault.
---

# Microsoft Agent Framework - Azure RBAC

## Roles
- Azure AI Developer or equivalent for Agents Service
- Key Vault Secrets User
- Storage Blob Data Reader (if using blob storage)

## az CLI Examples

```bash
az role assignment create --assignee $APP_ID \
  --role "Key Vault Secrets User" --scope $KEYVAULT_SCOPE
```

