"""Optional asynchronous transport around the existing Cobot RTC/HIL runtime."""
import online_runtime
from async_remote import AsyncRemote

class AsyncRuntime(AsyncRemote,online_runtime.OnlineRuntime):
    pass

if __name__=='__main__':
    online_runtime.OnlineRuntime=AsyncRuntime
    online_runtime.main()
