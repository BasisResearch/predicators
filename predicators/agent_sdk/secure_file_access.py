"""Secure file access utilities for Claude SDK tools.

Provides path validation and sandboxing to prevent agents from accessing
sensitive files like .env, credentials, or files outside the workspace.
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Set


class FileAccessValidator:
    """Validates file paths to prevent access to sensitive files."""

    # Default patterns for sensitive files to block
    DEFAULT_BLOCKED_PATTERNS = [
        r"\.env",  # Environment files
        r"\.env\.",  # .env.local, .env.production, etc.
        r"\.git/",  # Git internals
        r"\.git$",  # Git directory
        r"\.ssh/",  # SSH keys
        r"\.aws/",  # AWS credentials
        r"\.config/",  # Config files that might have credentials
        r"credentials",  # Any credentials file
        r"secret",  # Any secret file
        r"password",  # Any password file
        r"token",  # Any token file
        r"api[_-]?key",  # API keys
        r"\.pem$",  # Certificate files
        r"\.key$",  # Key files
        r"\.cert$",  # Certificate files
        r"id_rsa",  # SSH private keys
        r"id_ecdsa",  # SSH private keys
        r"id_ed25519",  # SSH private keys
        r"\.pypirc$",  # PyPI credentials
        r"\.netrc$",  # Network credentials
        r"\.npmrc$",  # NPM credentials
        r"\.dockercfg$",  # Docker credentials
        r"docker-compose\.yml$",  # May contain secrets
        r"\.kube/",  # Kubernetes config
        r"\.gnupg/",  # GPG keys
    ]

    def __init__(
        self,
        workspace_root: str,
        allowed_dirs: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        allow_hidden_files: bool = False,
    ):
        """Initialize the file access validator.

        Args:
            workspace_root: The root directory of the workspace (absolute path).
            allowed_dirs: List of allowed subdirectories (relative to workspace_root).
                         If None, allows access to entire workspace except blocked patterns.
                         Example: ["predicators", "scripts", "tests"]
            blocked_patterns: Additional regex patterns for files to block.
                            Defaults to DEFAULT_BLOCKED_PATTERNS.
            allow_hidden_files: Whether to allow access to hidden files (starting with .).
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_dirs = ([self.workspace_root / d
                              for d in allowed_dirs] if allowed_dirs else None)
        self.blocked_patterns = (blocked_patterns
                                 if blocked_patterns is not None else
                                 self.DEFAULT_BLOCKED_PATTERNS.copy())
        self.allow_hidden_files = allow_hidden_files

        # Compile regex patterns for efficiency
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.blocked_patterns
        ]

    def validate_path(self,
                      file_path: str,
                      operation: str = "read") -> tuple[bool, str]:
        """Validate if a file path is safe to access.

        Args:
            file_path: The file path to validate (can be relative or absolute).
            operation: The operation to perform ("read" or "write").

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        try:
            # Resolve to absolute path
            abs_path = Path(file_path).resolve()
        except Exception as e:
            return False, f"Invalid path: {e}"

        # Check if path is within workspace
        try:
            abs_path.relative_to(self.workspace_root)
        except ValueError:
            return False, f"Access denied: Path outside workspace root: {abs_path}"

        # Check if path is within allowed directories
        if self.allowed_dirs is not None:
            is_in_allowed = False
            for allowed_dir in self.allowed_dirs:
                try:
                    abs_path.relative_to(allowed_dir)
                    is_in_allowed = True
                    break
                except ValueError:
                    continue
            if not is_in_allowed:
                return False, f"Access denied: Path not in allowed directories: {abs_path}"

        # Check for hidden files
        if not self.allow_hidden_files:
            for part in abs_path.parts:
                if part.startswith(".") and part not in [".", ".."]:
                    return False, f"Access denied: Hidden file/directory: {abs_path}"

        # Check against blocked patterns
        path_str = str(abs_path)
        for pattern in self._compiled_patterns:
            if pattern.search(path_str):
                return False, f"Access denied: Path matches blocked pattern '{pattern.pattern}': {abs_path}"

        # Additional write validation
        if operation == "write":
            # Prevent writing to critical files
            if abs_path.name in [
                    "setup.py", "requirements.txt", "pyproject.toml"
            ]:
                return False, f"Access denied: Cannot modify critical file: {abs_path}"

        return True, ""

    def get_safe_path(self,
                      file_path: str,
                      operation: str = "read") -> Optional[Path]:
        """Get a validated absolute path, or None if invalid.

        Args:
            file_path: The file path to validate.
            operation: The operation to perform ("read" or "write").

        Returns:
            Absolute Path object if valid, None otherwise.
        """
        is_valid, error = self.validate_path(file_path, operation)
        if is_valid:
            return Path(file_path).resolve()
        return None

    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a new pattern to the blocked list.

        Args:
            pattern: Regex pattern to block.
        """
        self.blocked_patterns.append(pattern)
        self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))


def create_secure_file_tools(validator: FileAccessValidator) -> list:
    """Create secure file access tools with path validation.

    These tools wrap basic file operations with security checks.

    Args:
        validator: FileAccessValidator instance to use for validation.

    Returns:
        List of MCP tool functions.
    """
    from claude_agent_sdk import tool

    @tool(
        "secure_read_file",
        "Read contents of a file within the workspace (with security restrictions)",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description":
                    "Path to the file (relative to workspace root)"
                }
            },
            "required": ["file_path"]
        })
    async def secure_read_file(args: dict) -> dict:
        file_path = args["file_path"]

        is_valid, error = validator.validate_path(file_path, "read")
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {error}"
                }],
                "is_error": True
            }

        safe_path = validator.get_safe_path(file_path, "read")
        if safe_path is None or not safe_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: File not found: {file_path}"
                }],
                "is_error":
                True
            }

        try:
            content = safe_path.read_text(encoding="utf-8")
            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error reading file: {e}"
                }],
                "is_error": True
            }

    @tool(
        "secure_list_directory",
        "List contents of a directory within the workspace (with security restrictions)",
        {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type":
                    "string",
                    "description":
                    "Path to the directory (relative to workspace root)"
                }
            },
            "required": ["dir_path"]
        })
    async def secure_list_directory(args: dict) -> dict:
        dir_path = args["dir_path"]

        is_valid, error = validator.validate_path(dir_path, "read")
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {error}"
                }],
                "is_error": True
            }

        safe_path = validator.get_safe_path(dir_path, "read")
        if safe_path is None or not safe_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Directory not found: {dir_path}"
                }],
                "is_error":
                True
            }

        if not safe_path.is_dir():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Not a directory: {dir_path}"
                }],
                "is_error":
                True
            }

        try:
            entries = []
            for item in sorted(safe_path.iterdir()):
                # Filter out blocked items
                item_is_valid, _ = validator.validate_path(str(item), "read")
                if item_is_valid:
                    rel_path = item.relative_to(validator.workspace_root)
                    entries.append(
                        f"{'[DIR]' if item.is_dir() else '[FILE]'} {rel_path}")

            result = "\n".join(
                entries) if entries else "(empty or no accessible files)"
            return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error listing directory: {e}"
                }],
                "is_error":
                True
            }

    @tool(
        "secure_write_file",
        "Write contents to a file within the workspace (with security restrictions)",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description":
                    "Path to the file (relative to workspace root)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    async def secure_write_file(args: dict) -> dict:
        file_path = args["file_path"]
        content = args["content"]

        is_valid, error = validator.validate_path(file_path, "write")
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {error}"
                }],
                "is_error": True
            }

        safe_path = validator.get_safe_path(file_path, "write")
        if safe_path is None:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Invalid path: {file_path}"
                }],
                "is_error":
                True
            }

        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding="utf-8")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully wrote to {file_path}"
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error writing file: {e}"
                }],
                "is_error": True
            }

    return [secure_read_file, secure_list_directory, secure_write_file]


def create_validator_for_predicators_workspace(
        workspace_root: str,
        restrict_to_results: bool = True) -> FileAccessValidator:
    """Create a validator configured for the predicators workspace.

    Args:
        workspace_root: Path to the predicators workspace root.
        restrict_to_results: If True, only allow access to results/logs directories.

    Returns:
        Configured FileAccessValidator.
    """
    if restrict_to_results:
        # Most restrictive: only results and logs
        allowed_dirs = [
            "results", "logs", "saved_approaches", "saved_datasets"
        ]
    else:
        # Allow access to source and test directories, but not sensitive areas
        allowed_dirs = ["predicators", "tests", "scripts", "results", "logs"]

    # Add predicators-specific blocked patterns
    additional_blocks = [
        r"machines\.txt$",  # Cluster machine list
        r"\.pkl$",  # Pickle files might contain sensitive data
    ]

    validator = FileAccessValidator(
        workspace_root=workspace_root,
        allowed_dirs=allowed_dirs,
        blocked_patterns=FileAccessValidator.DEFAULT_BLOCKED_PATTERNS +
        additional_blocks,
        allow_hidden_files=False,
    )

    return validator
