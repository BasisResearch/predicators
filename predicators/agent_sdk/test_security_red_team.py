"""Red-teaming script to test security of file access validator.

This script attempts various attacks to access sensitive files
and verifies that the FileAccessValidator blocks them appropriately.

Run with:
    python -m predicators.agent_sdk.test_security_red_team
"""
import os
import tempfile
from pathlib import Path

from predicators.agent_sdk.secure_file_access import FileAccessValidator


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def test_case(name: str, validator: FileAccessValidator, path: str, should_block: bool) -> bool:
    """Test a single path and verify it's blocked/allowed as expected.

    Returns:
        True if test passes, False otherwise.
    """
    is_valid, error = validator.validate_path(path, "read")
    
    if should_block:
        # Expected to be blocked
        if not is_valid:
            print(f"  {Colors.GREEN}✓ PASS{Colors.ENDC}: Correctly blocked: {path}")
            print(f"         Reason: {error}")
            return True
        else:
            print(f"  {Colors.RED}✗ FAIL{Colors.ENDC}: {Colors.BOLD}SECURITY LEAK!{Colors.ENDC} Should have blocked: {path}")
            return False
    else:
        # Expected to be allowed
        if is_valid:
            print(f"  {Colors.GREEN}✓ PASS{Colors.ENDC}: Correctly allowed: {path}")
            return True
        else:
            print(f"  {Colors.RED}✗ FAIL{Colors.ENDC}: Incorrectly blocked: {path}")
            print(f"         Reason: {error}")
            return False


def run_red_team_tests():
    """Run comprehensive red-teaming tests."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== FILE ACCESS SECURITY RED-TEAMING ==={Colors.ENDC}\n")
    
    # Create a temporary workspace for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Create test directory structure
        (workspace / "results").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "predicators").mkdir()
        (workspace / ".git").mkdir()
        (workspace / ".env").write_text("SECRET_KEY=12345")
        (workspace / "credentials.txt").write_text("password123")
        (workspace / "results" / "output.txt").write_text("Safe file")
        
        # Create validator - most restrictive mode
        validator = FileAccessValidator(
            workspace_root=str(workspace),
            allowed_dirs=["results", "logs"],
            allow_hidden_files=False,
        )
        
        test_results = []
        
        print(f"{Colors.BOLD}Test Category: Environment Files{Colors.ENDC}")
        test_results.append(test_case("Env file 1", validator, str(workspace / ".env"), should_block=True))
        test_results.append(test_case("Env file 2", validator, str(workspace / ".env.local"), should_block=True))
        test_results.append(test_case("Env file 3", validator, str(workspace / ".env.production"), should_block=True))
        test_results.append(test_case("Env file 4", validator, str(workspace / "config" / ".env"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Credential Files{Colors.ENDC}")
        test_results.append(test_case("Credentials", validator, str(workspace / "credentials.txt"), should_block=True))
        test_results.append(test_case("Secrets", validator, str(workspace / "secrets.json"), should_block=True))
        test_results.append(test_case("AWS creds", validator, str(workspace / ".aws" / "credentials"), should_block=True))
        test_results.append(test_case("SSH key", validator, str(workspace / ".ssh" / "id_rsa"), should_block=True))
        test_results.append(test_case("API key", validator, str(workspace / "api_key.txt"), should_block=True))
        test_results.append(test_case("Token", validator, str(workspace / "auth_token.json"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Git Internals{Colors.ENDC}")
        test_results.append(test_case("Git dir", validator, str(workspace / ".git"), should_block=True))
        test_results.append(test_case("Git config", validator, str(workspace / ".git" / "config"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Hidden Files{Colors.ENDC}")
        test_results.append(test_case("Hidden file", validator, str(workspace / ".hidden"), should_block=True))
        test_results.append(test_case("DS_Store", validator, str(workspace / ".DS_Store"), should_block=True))
        test_results.append(test_case("Vim swap", validator, str(workspace / ".vimrc"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Path Traversal Attacks{Colors.ENDC}")
        test_results.append(test_case("Parent dir 1", validator, str(workspace.parent / "etc" / "passwd"), should_block=True))
        test_results.append(test_case("Parent dir 2", validator, str(workspace / ".." / ".env"), should_block=True))
        test_results.append(test_case("Absolute path", validator, "/etc/passwd", should_block=True))
        test_results.append(test_case("Home dir", validator, os.path.expanduser("~/.bash_history"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Directory Restrictions{Colors.ENDC}")
        test_results.append(test_case("Outside allowed", validator, str(workspace / "predicators" / "file.py"), should_block=True))
        test_results.append(test_case("In results", validator, str(workspace / "results" / "output.txt"), should_block=False))
        test_results.append(test_case("In logs", validator, str(workspace / "logs" / "log.txt"), should_block=False))
        
        print(f"\n{Colors.BOLD}Test Category: Certificate/Key Files{Colors.ENDC}")
        test_results.append(test_case("PEM file", validator, str(workspace / "cert.pem"), should_block=True))
        test_results.append(test_case("Key file", validator, str(workspace / "private.key"), should_block=True))
        test_results.append(test_case("Cert file", validator, str(workspace / "server.cert"), should_block=True))
        
        print(f"\n{Colors.BOLD}Test Category: Package Manager Credentials{Colors.ENDC}")
        test_results.append(test_case("PyPI", validator, str(workspace / ".pypirc"), should_block=True))
        test_results.append(test_case("NPM", validator, str(workspace / ".npmrc"), should_block=True))
        test_results.append(test_case("Netrc", validator, str(workspace / ".netrc"), should_block=True))
        
        # Summary
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}")
        passed = sum(test_results)
        total = len(test_results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        if passed == total:
            print(f"{Colors.BOLD}{Colors.GREEN}ALL TESTS PASSED!{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ {passed}/{total} security checks passed ({percentage:.1f}%){Colors.ENDC}")
            print(f"\n{Colors.GREEN}🛡️  The validator successfully blocked all attack attempts!{Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}SECURITY ISSUES FOUND!{Colors.ENDC}")
            print(f"{Colors.RED}✗ {total - passed}/{total} security checks FAILED ({100-percentage:.1f}%){Colors.ENDC}")
            print(f"{Colors.YELLOW}⚠️  Please review the failed tests above to identify security leaks.{Colors.ENDC}")
        
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}\n")
        
        return passed == total


def demonstrate_attacks():
    """Demonstrate example attack scenarios."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXAMPLE ATTACK SCENARIOS ==={Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Scenario 1: Agent tries to read .env file{Colors.ENDC}")
    print("Agent: I'll check if there's a .env file with API keys...")
    print("Agent: secure_read_file(file_path='.env')")
    print(f"{Colors.RED}Blocked!{Colors.ENDC} Reason: Path matches blocked pattern")
    
    print(f"\n{Colors.BOLD}Scenario 2: Agent tries path traversal{Colors.ENDC}")
    print("Agent: Let me check the parent directory...")
    print("Agent: secure_read_file(file_path='../../etc/passwd')")
    print(f"{Colors.RED}Blocked!{Colors.ENDC} Reason: Path outside workspace root")
    
    print(f"\n{Colors.BOLD}Scenario 3: Agent tries to access SSH keys{Colors.ENDC}")
    print("Agent: I'll look for authentication credentials...")
    print("Agent: secure_read_file(file_path='~/.ssh/id_rsa')")
    print(f"{Colors.RED}Blocked!{Colors.ENDC} Reason: Path outside workspace root")
    
    print(f"\n{Colors.BOLD}Scenario 4: Agent tries to read git config{Colors.ENDC}")
    print("Agent: Let me check the repository configuration...")
    print("Agent: secure_read_file(file_path='.git/config')")
    print(f"{Colors.RED}Blocked!{Colors.ENDC} Reason: Hidden file/directory")
    
    print(f"\n{Colors.BOLD}Scenario 5: Agent tries allowed file{Colors.ENDC}")
    print("Agent: Let me check the results...")
    print("Agent: secure_read_file(file_path='results/output.txt')")
    print(f"{Colors.GREEN}Allowed!{Colors.ENDC} Successfully read file content")
    
    print()


if __name__ == "__main__":
    demonstrate_attacks()
    success = run_red_team_tests()
    
    if not success:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  WARNING: Security vulnerabilities detected!{Colors.ENDC}")
        print(f"{Colors.YELLOW}Review the failed tests and update the FileAccessValidator.{Colors.ENDC}\n")
        exit(1)
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Security validation passed!{Colors.ENDC}")
        print(f"{Colors.GREEN}The FileAccessValidator is working correctly.{Colors.ENDC}\n")
        exit(0)
