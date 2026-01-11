#/storage/emulated/0/Vxt/Vxt/vera_xt/utils/error_handler.py
#!/usr/bin/env python3
"""
Error Handler - Modular error recovery system
If something fails, system continues working
"""

def safe_execute(func, fallback_func=None, *args, **kwargs):
    """Execute function safely, fallback if error"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  Component failed: {e}")
        if fallback_func:
            return fallback_func(*args, **kwargs)
        return None

def modular_component(func):
    """Decorator: if component fails, system continues"""
    def wrapper(*args, **kwargs):
        return safe_execute(
            func, 
            lambda *a, **kw: f"Component temporarily unavailable: {func.__name__}", 
            *args, **kwargs
        )
    return wrapper

def confirm_error_fix(error_msg: str) -> bool:
    """Ask user to confirm error handling approach"""
    print(f"\n❌ Error occurred: {error_msg}")
    response = input("Should I try to fix this automatically? (y/n): ").lower().strip()
    return response in ['y', 'yes']

# Example usage:
# @modular_component
# def risky_function():
#     # This function can fail without breaking the system
#     pass
