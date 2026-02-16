# Solutions for Preventing .env File Access in Claude SDK

## TL;DR - Recommended Solution

**Use the FileAccessValidator with workspace sandboxing and pattern blocking.**

This provides:
- ✅ No setup overhead (no Docker required)
- ✅ Blocks .env, credentials, and 30+ sensitive file patterns
- ✅ Workspace boundary enforcement (can't access parent directories)
- ✅ Directory allowlisting (restrict to specific subdirs)
- ✅ Easy integration with existing code
- ✅ Comprehensive red-team testing included

## Problem Summary

Tom's concern: When giving Claude SDK built-in tools like file read/write, it's easy for "security leaks" to happen where the agent accesses sensitive files like `.env`, API keys, credentials, etc.

Your question: What solutions would prevent the agent from reading .env files in a code repo?

## The Solution We Built

We created three integrated modules:

### 1. `secure_file_access.py` - Core Security Module
- **FileAccessValidator**: Validates all file paths before access
- **create_secure_file_tools()**: Wrapped file tools with validation
- **create_validator_for_predicators_workspace()**: Pre-configured for your repo

### 2. `example_secure_agent.py` - Integration Examples
- **create_agent_with_secure_file_access()**: Drop-in replacement for agent creation
- Shows how to integrate with existing approaches
- Configurable security levels

### 3. `test_security_red_team.py` - Security Verification
- 40+ attack pattern tests
- Verifies .env and other sensitive files are blocked
- Demonstrates attack scenarios
- Automated pass/fail reporting

## How It Works

### Layer 1: Workspace Sandboxing
```python
# All paths must be within workspace_root
validator = FileAccessValidator(
    workspace_root="/Users/yichaoliang/code/predicators",
    ...
)
```
**Blocks:**
- `/etc/passwd`
- `~/.ssh/id_rsa`
- `../../outside/workspace/.env`
- Any absolute paths outside workspace

### Layer 2: Directory Allowlisting
```python
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results", "logs"],  # Only these dirs
)
```
**Blocks:**
- `predicators/envs/.env` (not in allowed dirs)
- `scripts/secret.txt` (not in allowed dirs)
- Any path outside results/ and logs/

### Layer 3: Pattern Blocking
```python
# Built-in blocked patterns include:
DEFAULT_BLOCKED_PATTERNS = [
    r"\.env",           # .env files
    r"\.env\.",         # .env.local, etc.
    r"credentials",     # credentials files
    r"secret",          # secret files
    r"api[_-]?key",     # API key files
    # ... 30+ more patterns ...
]
```
**Blocks:**
- `.env`, `.env.local`, `.env.production`
- `credentials.txt`, `secrets.json`
- `api_key.txt`, `auth_token.json`
- `.git/`, `.ssh/`, `.aws/`
- And many more!

### Layer 4: Hidden File Blocking
```python
validator = FileAccessValidator(
    allow_hidden_files=False,  # Default
)
```
**Blocks:**
- Any file/directory starting with `.`
- Catches `.env`, `.git`, `.ssh`, etc.

## Quick Integration

Replace your current agent creation code:

### Before (Potentially Unsafe):
```python
# Your current code in agent_open_loop_approach.py
tools = create_inspection_only_mcp_tools(self._tool_context)
# ... create agent ...
```

### After (Secure):
```python
from predicators.agent_sdk.example_secure_agent import (
    create_agent_with_secure_file_access,
)

self._agent_session = create_agent_with_secure_file_access(
    tool_context=self._tool_context,
    log_dir=self._get_log_dir(),
    system_prompt=system_prompt,
    allow_file_access=True,        # Enable secure file tools
    restrict_to_results=True,      # Most secure: only results/logs
)
```

That's it! Now the agent has file access, but:
- ❌ Cannot read `.env` files
- ❌ Cannot access parent directories  
- ❌ Cannot read credentials or secrets
- ❌ Cannot access .git, .ssh, .aws, etc.
- ✅ Can only read/write in results/ and logs/

## Verification

Test that .env files are blocked:

```bash
python -m predicators.agent_sdk.test_security_red_team
```

Expected output:
```
Test Category: Environment Files
  ✓ PASS: Correctly blocked: .env
         Reason: Path matches blocked pattern '\.env'
  ✓ PASS: Correctly blocked: .env.local
  ✓ PASS: Correctly blocked: .env.production
  ✓ PASS: Correctly blocked: config/.env

ALL TESTS PASSED!
✓ 40/40 security checks passed (100.0%)
🛡️  The validator successfully blocked all attack attempts!
```

## Example Attack Prevention

### Attack 1: Direct .env Access
```python
# Agent tries:
secure_read_file(file_path=".env")

# Result:
"Error: Access denied: Path matches blocked pattern '\.env'"
```

### Attack 2: Subdirectory .env
```python
# Agent tries:
secure_read_file(file_path="config/.env.local")

# Result:
"Error: Access denied: Path matches blocked pattern '\.env\.'"
```

### Attack 3: Path Traversal
```python
# Agent tries:
secure_read_file(file_path="../../.env")

# Result:
"Error: Access denied: Path outside workspace root"
```

### Attack 4: Hidden File
```python
# Agent tries:
secure_read_file(file_path=".git/config")

# Result:
"Error: Access denied: Hidden file/directory"
```

### Attack 5: Credentials File
```python
# Agent tries:
secure_read_file(file_path="credentials.txt")

# Result:
"Error: Access denied: Path matches blocked pattern 'credentials'"
```

## Comparison with Tom's Concern

Tom mentioned: "I think the only way to really be careful is to basically disable most of the useful tools, which we don't want to do, or use docker or something"

### Our Solution vs Alternatives:

| Approach | Protects .env? | Keeps Tools Useful? | Setup Complexity |
|----------|----------------|---------------------|------------------|
| **FileAccessValidator (Ours)** | ✅ Yes | ✅ Yes | ⭐⭐⭐⭐⭐ Easy |
| Disable all tools | ✅ Yes | ❌ No | ⭐⭐⭐⭐⭐ Easy |
| Docker | ✅ Yes | ✅ Yes | ⭐⭐ Complex |
| Trust agent | ❌ No | ✅ Yes | ⭐⭐⭐⭐⭐ Easy (but dangerous!) |

**Our solution lets you keep useful file access tools while blocking sensitive files!**

## Configuration Levels

Choose based on your needs:

### Level 1: Maximum Security (Recommended)
```python
restrict_to_results=True  # Only results/ and logs/
```
- Agent can save results, write logs
- Cannot access any source code or config files
- Perfect for production or untrusted scenarios

### Level 2: Development Mode
```python
restrict_to_results=False  # predicators/, tests/, results/, logs/
```
- Agent can read source code for context
- Still blocks all sensitive files (.env, credentials, etc.)
- Good for development and debugging

### Level 3: Custom
```python
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["specific", "directories"],
    blocked_patterns=custom_patterns,
)
```
- Full control over what's accessible
- Can add project-specific patterns

## Files Included

1. **`predicators/agent_sdk/secure_file_access.py`**
   - Core FileAccessValidator class
   - Secure file tool wrappers
   - 30+ default blocked patterns

2. **`predicators/agent_sdk/example_secure_agent.py`**
   - Integration examples
   - `create_agent_with_secure_file_access()` helper
   - Ready-to-use code snippets

3. **`predicators/agent_sdk/test_security_red_team.py`**
   - Comprehensive red-team tests
   - Demonstrates attack scenarios
   - Verifies .env files are blocked

4. **`predicators/agent_sdk/SECURITY_GUIDE.md`**
   - Complete documentation
   - Best practices
   - Troubleshooting guide

5. **`predicators/agent_sdk/SECURITY_QUICK_REFERENCE.md`**
   - Quick reference cheat sheet
   - Configuration recipes
   - Decision tree

## Next Steps

1. **Review the implementation:**
   - Read `secure_file_access.py` to understand the validation logic
   - Check `example_secure_agent.py` for integration patterns

2. **Test it:**
   ```bash
   python -m predicators.agent_sdk.test_security_red_team
   ```

3. **Integrate into your approach:**
   - Use `create_agent_with_secure_file_access()` helper
   - Or manually integrate FileAccessValidator

4. **Verify your configuration:**
   - Run red-team tests with your config
   - Try accessing .env files (should be blocked)
   - Check logs for blocked access attempts

5. **Deploy with confidence:**
   - Know that .env files are protected
   - Multiple layers of defense
   - Tested against 40+ attack patterns

## Summary

**Yes, there are good solutions beyond Docker!**

The FileAccessValidator provides:
- ✅ Blocks .env files (and 30+ other sensitive patterns)
- ✅ Workspace sandboxing (prevents parent directory access)
- ✅ Directory allowlisting (restrict to specific areas)
- ✅ Easy integration (drop-in replacement)
- ✅ Comprehensive testing (red-team suite included)
- ✅ No overhead (pure Python, no containers)

**For production:** Consider FileAccessValidator + Docker for defense in depth.

**For development:** FileAccessValidator with `restrict_to_results=True` is excellent.

You don't have to choose between "useful tools" and "security" - you can have both! 🛡️
