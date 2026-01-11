#/storage/emulated/0/Vxt/Vxt/vera_xt/utils/file_operations.py
#!/usr/bin/env python3
"""
Safe File Operations - Sandboxed file handling
"""

import os
from pathlib import Path
from typing import Optional, Union

class SafeFileOperations:
    def __init__(self, workspace_dir: str = "Workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
    
    def safe_path(self, filepath: str) -> Optional[Path]:
        """Ensure filepath is within workspace"""
        path = Path(filepath)
        try:
            # Resolve the path and check if it's within workspace
            resolved_path = path.resolve()
            workspace_resolved = self.workspace_dir.resolve()
            
            if str(resolved_path).startswith(str(workspace_resolved)):
                return resolved_path
            else:
                print(f"❌ Security: Path {filepath} is outside workspace")
                return None
        except Exception:
            return None
    
    def write_file(self, filepath: str, content: str) -> bool:
        """Safely write file within workspace"""
        safe_path = self.safe_path(filepath)
        if safe_path:
            try:
                safe_path.parent.mkdir(parents=True, exist_ok=True)
                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Wrote to: {filepath}")
                return True
            except Exception as e:
                print(f"❌ Write failed: {e}")
                return False
        return False
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Safely read file within workspace"""
        safe_path = self.safe_path(filepath)
        if safe_path and safe_path.exists():
            try:
                with open(safe_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"❌ Read failed: {e}")
                return None
        else:
            print(f"❌ File not found: {filepath}")
            return None
    
    def list_files(self, directory: str = ".") -> list:
        """List files in workspace directory"""
        safe_path = self.safe_path(directory)
        if safe_path and safe_path.exists():
            try:
                return [f.name for f in safe_path.iterdir() if f.is_file()]
            except Exception as e:
                print(f"❌ List failed: {e}")
                return []
        return []
    
    def delete_file(self, filepath: str) -> bool:
        """Safely delete file within workspace"""
        safe_path = self.safe_path(filepath)
        if safe_path and safe_path.exists():
            try:
                safe_path.unlink()
                print(f"✅ Deleted: {filepath}")
                return True
            except Exception as e:
                print(f"❌ Delete failed: {e}")
                return False
        return False

# Global instance
file_ops = SafeFileOperations()
