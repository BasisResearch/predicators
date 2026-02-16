# Security Quick Reference

## 🚨 Common Attack Patterns (What Gets Blocked)

| Attack Type | Example | Blocked By |
|-------------|---------|------------|
| Environment files | `.env`, `.env.local` | Pattern matching |
| Credentials | `credentials.txt`, `secrets.json` | Pattern matching |
| SSH keys | `~/.ssh/id_rsa` | Workspace boundary + pattern |
| AWS creds | `~/.aws/credentials` | Workspace boundary + pattern |
| Git config | `.git/config` | Hidden file blocking |
| Path traversal | `../../etc/passwd` | Workspace boundary |
| Absolute paths | `/etc/passwd` | Workspace boundary |
| API keys | `api_key.txt`, `token.json` | Pattern matching |
| Certificates | `cert.pem`, `private.key` | Pattern matching |
| Package creds | `.pypirc`, `.npmrc` | Pattern matching |

## 🔒 Security Levels (Choose One)

### Level 1: Maximum Security (Recommended for Production)
```python
validator = create_validator_for_predicators_workspace(
    workspace_root=workspace_root,
    restrict_to_results=True,  # Only results/ and logs/
)
```
**Pros:** Minimal attack surface  
**Cons:** Limited functionality  
**Use When:** Production, untrusted agents, sensitive data

### Level 2: Balanced Security (Recommended for Development)
```python
validator = create_validator_for_predicators_workspace(
    workspace_root=workspace_root,
    restrict_to_results=False,  # predicators/, tests/, results/, logs/
)
```
**Pros:** Can read code for debugging  
**Cons:** More attack surface  
**Use When:** Development, trusted environments

### Level 3: Custom Security
```python
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["specific", "dirs", "only"],
    blocked_patterns=FileAccessValidator.DEFAULT_BLOCKED_PATTERNS + [
        r"my_custom_pattern",
    ],
    allow_hidden_files=False,
)
```
**Pros:** Fine-grained control  
**Cons:** Requires careful configuration  
**Use When:** Specialized requirements

### Level 4: No File Access (Maximum Security)
```python
# Only use inspection tools, no file access
tools = create_inspection_only_mcp_tools(tool_context)
# Don't add any file tools
```
**Pros:** Zero file access risk  
**Cons:** No file access capability  
**Use When:** Don't need file access

## ⚡ Quick Setup Recipes

### Recipe 1: Secure Open-Loop Agent
```python
from predicators.agent_sdk.example_secure_agent import (
    create_agent_with_secure_file_access,
)

agent = create_agent_with_secure_file_access(
    tool_context=tool_context,
    log_dir=log_dir,
    system_prompt=system_prompt,
    allow_file_access=True,
    restrict_to_results=True,
)
```

### Recipe 2: Results-Only Access
```python
from predicators.agent_sdk.secure_file_access import (
    FileAccessValidator,
    create_secure_file_tools,
)

validator = FileAccessValidator(
    workspace_root="/path/to/workspace",
    allowed_dirs=["results", "logs"],
)
file_tools = create_secure_file_tools(validator)
```

### Recipe 3: Read-Only Access
```python
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results", "data"],
)
# Only create read tools
from claude_agent_sdk import tool

@tool("secure_read_only", "Read file (read-only)", {...})
async def secure_read(args):
    is_valid, error = validator.validate_path(args["path"], "read")
    if not is_valid:
        return {"content": [{"type": "text", "text": error}], "is_error": True}
    # ... read file ...
```

## 🧪 Testing Checklist

Before deploying:
- [ ] Run red-team tests: `python -m predicators.agent_sdk.test_security_red_team`
- [ ] Verify only intended directories accessible
- [ ] Test with actual sensitive files (safe copy)
- [ ] Check logs for blocked access attempts
- [ ] Review allowed_tools list
- [ ] Verify no hardcoded secrets in accessible files

## 🚩 Red Flags (Security Smells)

❌ **Dangerous Patterns:**
```python
# NO: Allowing everything
allowed_dirs=None  

# NO: Allowing hidden files
allow_hidden_files=True  

# NO: Empty blocked patterns
blocked_patterns=[]  

# NO: Bypassing validation
validator.validate_path = lambda *args: (True, "")  

# NO: Trusting user input directly
path = user_input  # Don't use directly!
```

✅ **Safe Patterns:**
```python
# YES: Specific allowlist
allowed_dirs=["results", "logs"]

# YES: Block hidden files
allow_hidden_files=False

# YES: Use default + custom blocks
blocked_patterns=FileAccessValidator.DEFAULT_BLOCKED_PATTERNS + [...]

# YES: Always validate
is_valid, error = validator.validate_path(path, operation)
if not is_valid:
    return error_response
```

## 📋 Configuration Examples

### Minimal (Most Secure)
```python
FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results"],  # Single directory
    allow_hidden_files=False,
)
```

### Standard (Balanced)
```python
FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results", "logs", "data"],
    allow_hidden_files=False,
)
```

### Development (More Permissive)
```python
FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["predicators", "tests", "results", "logs"],
    allow_hidden_files=False,  # Still block hidden files!
)
```

### Custom (Application-Specific)
```python
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results", "logs"],
    allow_hidden_files=False,
)
# Add custom blocks
validator.add_blocked_pattern(r"experiment_\d+/private/")
validator.add_blocked_pattern(r"internal_notes\.txt$")
```

## 🔍 Debugging Access Issues

### Agent says "Access Denied"

1. **Check the error message** - it tells you why:
   - "Path outside workspace" → trying to access parent dirs
   - "Path matches blocked pattern" → hitting a blocklist
   - "Not in allowed directories" → directory not in allowlist
   - "Hidden file/directory" → starts with `.`

2. **Verify your configuration:**
   ```python
   # Print validator settings
   print(f"Workspace root: {validator.workspace_root}")
   print(f"Allowed dirs: {validator.allowed_dirs}")
   print(f"Blocked patterns: {validator.blocked_patterns}")
   
   # Test a specific path
   is_valid, error = validator.validate_path("path/to/test", "read")
   print(f"Valid: {is_valid}, Error: {error}")
   ```

3. **Solutions:**
   - Move file to allowed directory
   - Add directory to `allowed_dirs`
   - Rename file if hitting blocked pattern
   - Review if you really need access (security first!)

## 📊 Security Comparison

| Method | Security | Setup | Performance | Flexibility |
|--------|----------|-------|-------------|-------------|
| No Tools | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| FileAccessValidator | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Docker Container | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| VM/Sandbox | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| Trust Agent | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation:** Start with FileAccessValidator + restrictive config, add Docker for production.

## 🎯 Decision Tree

```
Need file access?
├─ NO → Use inspection-only tools ✅
└─ YES → Need to read code?
    ├─ NO → restrict_to_results=True ✅
    └─ YES → Need to write files?
        ├─ NO → Read-only validator ✅
        └─ YES → Is this production?
            ├─ YES → Docker + validator ✅
            └─ NO → Full validator + testing ✅
```

## 📞 Emergency Response

If you suspect a security breach:

1. **Immediately:** Rotate all credentials/API keys
2. **Check logs:** Review agent activity logs
3. **Audit files:** Check for unauthorized modifications
4. **Review config:** Verify validator settings
5. **Run red-team:** `python -m predicators.agent_sdk.test_security_red_team`
6. **Tighten security:** Reduce allowed_dirs, add more blocks

## 💡 Pro Tips

1. **Start restrictive, gradually open:** Easier to loosen than tighten
2. **Log everything:** Monitor agent file access patterns
3. **Regular audits:** Review security monthly
4. **Defense in depth:** Use multiple layers (validator + Docker + monitoring)
5. **Test before deploy:** Always run red-team suite
6. **Document exceptions:** If you bypass security, document why
7. **Environment variables:** Keep secrets out of files
8. **Git secrets scanning:** Use tools like `git-secrets`, `truffleHog`

---

**Remember:** When in doubt, block it out! 🛡️
