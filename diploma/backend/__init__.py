try:
    from .service import AppService
    print("✅ Backend: AppService imported")
except ImportError as e:
    print(f"❌ Backend import error: {e}")
    class AppService: pass  # заглушка
    
print("✅ Backend module ready")
