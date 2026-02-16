# Security Guide: Preventing File Access Leaks in Claude SDK

## Problem

When giving the Claude SDK agent built-in tools (like `read_file`, `write_file`), the agent can potentially access **any file on your system**, including:
- `.env` files with API keys
- SSH private keys (`~/.ssh/id_rsa`)
- AWS credentials (`~/.aws/credentials`)
- Database passwords
- Git configuration
- Other sensitive information

## Solution Overview

We provide a **layered security approach** with three main components:

1. **FileAccessValidator** - Path validation with allowlists/blocklists
2. **Secure File Tools** - Wrapped file operations with validation
3. **Red-Team Testing** - Automated security verification

## Quick Start

### Option 1: Most Secure (Recommended)
Restrict file access to only `results/` and `logs/` directories:

```python
from predicators.agent_sdk.secure_file_access import (
    create_validator_for_predicators_workspace,
    create_secure_file_tools,
)

# Create validator - only results/logs allowed
validator = create_validator_for_predicators_workspace(
    workspace_root="/path/to/predicators",
    restrict_to_results=True,  # Most secure
)

# Create secure tools
file_tools = create_secure_file_tools(validator)

# Add to your MCP server
tools = create_inspection_only_mcp_tools(tool_context) + file_tools
```

### Option 2: Moderate Security
Allow access to source code but block sensitive files:

```python
validator = create_validator_for_predicators_workspace(
    workspace_root="/path/to/predicators",
    restrict_to_results=False,  # Allow predicators/, tests/, etc.
)
```

### Option 3: Custom Configuration
Fine-grained control:

```python
from predicators.agent_sdk.secure_file_access import FileAccessValidator

validator = FileAccessValidator(
    workspace_root="/path/to/predicators",
    allowed_dirs=["predicators/envs", "results", "logs"],  # Specific dirs
    allow_hidden_files=False,  # Block .git, .env, etc.
)

# Add custom blocks
validator.add_blocked_pattern(r"my_secret_dir/")
validator.add_blocked_pattern(r"internal_config\.yaml$")
```

## Security Features

### 1. Workspace Sandboxing
- All paths must be within the workspace root
- Prevents access to `/etc/passwd`, `~/.ssh/`, etc.
- Blocks path traversal attacks (`../../sensitive`)

### 2. Directory Allowlisting
- Restrict to specific subdirectories (e.g., only `results/`)
- Prevents access to source code if not needed

### 3. Pattern Blocking
Blocks files matching sensitive patterns:
- Environment files: `.env`, `.env.local`, `.env.production`
- Credentials: `credentials`, `secrets`, `password`, `token`, `api_key`
- Keys: `.pem`, `.key`, `.cert`, `id_rsa`, SSH keys
- Git internals: `.git/`
- Package credentials: `.pypirc`, `.npmrc`, `.netrc`
- Config directories: `.ssh/`, `.aws/`, `.kube/`
- And more...

### 4. Hidden File Blocking
- Blocks access to files/dirs starting with `.`
- Prevents `.env`, `.git`, `.ssh` access

### 5. Write Protection
- Prevents modification of critical files (`setup.py`, etc.)
- All write operations validated the same as reads

## Integration Examples

### Example 1: Secure Agent Open-Loop Approach

```python
# In your approach __init__:
from predicators.agent_sdk.example_secure_agent import (
    create_agent_with_secure_file_access,
)

self._agent_session = create_agent_with_secure_file_access(
    tool_context=self._tool_context,
    log_dir="logs/agent_open_loop",
    system_prompt=system_prompt,
    allow_file_access=True,       # Enable file tools
    restrict_to_results=True,      # Most secure mode
)
```

### Example 2: Manual Integration

```python
from claude_agent_sdk import create_sdk_mcp_server
from predicators.agent_sdk.secure_file_access import (
    FileAccessValidator,
    create_secure_file_tools,
)
from predicators.agent_sdk.tools import create_inspection_only_mcp_tools

# Create validator
validator = FileAccessValidator(
    workspace_root="/path/to/workspace",
    allowed_dirs=["results", "logs"],
    allow_hidden_files=False,
)

# Create all tools
inspection_tools = create_inspection_only_mcp_tools(tool_context)
file_tools = create_secure_file_tools(validator)
all_tools = inspection_tools + file_tools

# Create MCP server
mcp_server = create_sdk_mcp_server(
    name="predicator_tools",
    version="1.0.0",
    tools=all_tools,
)

# Configure allowed tools
tool_prefix = "mcp__predicator_tools__"
allowed_tools = [
    f"{tool_prefix}inspect_types",
    f"{tool_prefix}inspect_predicates",
    f"{tool_prefix}inspect_options",
    f"{tool_prefix}inspect_trajectories",
    f"{tool_prefix}inspect_train_tasks",
    f"{tool_prefix}secure_read_file",        # Secure file access
    f"{tool_prefix}secure_list_directory",   # Secure directory listing
    f"{tool_prefix}secure_write_file",       # Secure file writing
]

# Create session manager
agent_session = AgentSessionManager(
    system_prompt=system_prompt,
    mcp_server=mcp_server,
    log_dir=log_dir,
    model_name=model_name,
    allowed_tools=allowed_tools,
)
```

## Red-Team Testing

**Always test your security configuration!**

Run the red-team test suite:

```bash
python -m predicators.agent_sdk.test_security_red_team
```

This will:
1. Demonstrate example attack scenarios
2. Run comprehensive security tests (40+ attack patterns)
3. Verify the validator blocks all malicious access
4. Report any security vulnerabilities

Example output:
```
=== EXAMPLE ATTACK SCENARIOS ===

Scenario 1: Agent tries to read .env file
Agent: secure_read_file(file_path='.env')
Blocked! Reason: Path matches blocked pattern

=== FILE ACCESS SECURITY RED-TEAMING ===

Test Category: Environment Files
  ✓ PASS: Correctly blocked: .env
  ✓ PASS: Correctly blocked: .env.local
  
Test Category: Credential Files
  ✓ PASS: Correctly blocked: credentials.txt
  ✓ PASS: Correctly blocked: .aws/credentials
  
... [40+ tests] ...

ALL TESTS PASSED!
✓ 40/40 security checks passed (100.0%)
🛡️  The validator successfully blocked all attack attempts!
```

## What Gets Blocked?

### Environment & Secrets
- `.env`, `.env.local`, `.env.production`
- `credentials.txt`, `secrets.json`
- `password`, `token`, `api_key` files

### Authentication
- `.ssh/id_rsa`, `.ssh/id_ecdsa` (SSH keys)
- `.aws/credentials` (AWS credentials)
- `.kube/config` (Kubernetes config)
- `.pypirc`, `.npmrc`, `.netrc` (package manager creds)

### Certificates & Keys  
- `*.pem`, `*.key`, `*.cert`
- `.gnupg/` (GPG keys)

### Source Control
- `.git/` directory and contents
- `.gitconfig`

### Docker
- `.dockercfg`, `docker-compose.yml` (may contain secrets)

### Any Hidden Files
- All files/directories starting with `.` (unless explicitly enabled)

## Best Practices

### 1. Principle of Least Privilege
Only grant the minimum access needed:
```python
# Good: Only results directory
allowed_dirs=["results"]

# Bad: Everything
allowed_dirs=None  # Careful!
```

### 2. Layer Security Measures
- Use workspace sandboxing
- Add directory restrictions
- Enable pattern blocking
- Disable hidden file access
- Add custom patterns for your use case

### 3. Regular Red-Team Testing
```bash
# Test after any security changes
python -m predicators.agent_sdk.test_security_red_team

# Add custom tests for your specific files
```

### 4. Audit Tool Access
Review what tools you're giving the agent:
```python
# Good: Explicit allowlist
allowed_tools = [
    "mcp__predicator_tools__secure_read_file",  # Controlled
]

# Bad: Allow all tools
permission_mode="bypassPermissions"  # Only with careful tool design!
```

### 5. Monitor Agent Activity
Log all file access attempts:
```python
# The secure tools automatically log blocked access
# Check logs for suspicious patterns
```

### 6. Use Environment Variables
Never hardcode sensitive values in files the agent can access:
```python
# Good
api_key = os.environ["API_KEY"]

# Bad - if agent can read this file
api_key = "sk-1234567890abcdef"  # DON'T DO THIS
```

## Advanced: Custom Validators

For specialized security needs:

```python
class CustomValidator(FileAccessValidator):
    def validate_path(self, file_path: str, operation: str):
        # Add custom validation logic
        is_valid, error = super().validate_path(file_path, operation)
        
        if not is_valid:
            return is_valid, error
            
        # Add your custom checks
        path = Path(file_path).resolve()
        
        # Example: Block files larger than 10MB
        if path.exists() and path.stat().st_size > 10 * 1024 * 1024:
            return False, "File too large"
            
        # Example: Block binary files for reads
        if operation == "read":
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return False, "Binary files not allowed"
                
        return True, ""
```

## Comparison with Other Solutions

| Solution | Pros | Cons |
|----------|------|------|
| **FileAccessValidator (Ours)** | No setup, integrated, fast | Application-level only |
| **Docker Container** | Strong isolation | Complex setup, slower |
| **VM/Sandbox** | Maximum isolation | Heavy overhead |
| **No File Tools** | Perfectly safe | Limited functionality |
| **Trust Agent** | Zero overhead | **DANGEROUS - Don't do this!** |

## Recommendations

1. **For Development/Testing**: Use `restrict_to_results=True`
2. **For Production**: Docker + FileAccessValidator (defense in depth)
3. **High Security**: No file tools at all, only custom domain tools
4. **Maximum Security**: Run in isolated container with no network access

## Troubleshooting

### Agent Can't Access Needed Files
```python
# Check validator configuration
validator = FileAccessValidator(
    workspace_root=workspace_root,
    allowed_dirs=["results", "logs", "data"],  # Add needed dirs
    allow_hidden_files=False,
)

# Or check blocked patterns
validator = create_validator_for_predicators_workspace(...)
# Remove overly restrictive patterns if needed (carefully!)
```

### False Positives
```python
# If legitimate files are blocked, you can:
# 1. Rename the file to not match blocked patterns
# 2. Move to an allowed directory
# 3. Customize the validator (carefully review security implications)
```

### Testing Custom Validators
```python
# Add test cases to test_security_red_team.py
test_results.append(test_case(
    "My custom file",
    validator,
    "path/to/my/file.txt",
    should_block=True,  # or False
))
```

## Summary

✅ **Use FileAccessValidator** - Always wrap file access tools  
✅ **Test with Red-Team Suite** - Verify your configuration  
✅ **Principle of Least Privilege** - Minimal access needed  
✅ **Monitor & Audit** - Check logs for suspicious activity  
✅ **Consider Docker** - For production deployments  

❌ **Never trust the agent** - Always validate  
❌ **Never bypass security** - "Just for testing" becomes permanent  
❌ **Never hardcode secrets** - In files the agent can access  

---

**Remember**: Security is about layers. Use multiple defenses and regularly test them!
