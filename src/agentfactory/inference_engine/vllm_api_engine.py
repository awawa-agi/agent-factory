import time
from pathlib import Path
from typing import Optional, Dict, Any, Set, Tuple
from requests_futures.sessions import FuturesSession
from concurrent.futures import as_completed

from .base_inference_engine import BaseInferenceEngine
from ..hparams import InferenceEngineConfig

class VLLMApiEngine(BaseInferenceEngine):
    def __init__(self, config: InferenceEngineConfig):
        super().__init__(config)

        self._server_urls = [url.rstrip('/') for url in config.urls]
        self.session = FuturesSession(max_workers=len(self._server_urls))

        self._wait_for_servers_ready(self.config.server_startup_timeout)
    
    def _send_parallel_requests(
        self, 
        endpoint: str, 
        method: str = "POST", 
        payload: Optional[Dict] = None,
        timeout: Optional[float] = None
    ) -> Tuple[Set[str], Dict[str, str]]:
        """Send requests to all servers and return (success_urls, failed_details)"""
        future_to_url = {}
        timeout_value = timeout or self.config.api_request_timeout
        
        for url in self._server_urls:
            full_url = f"{url}/{endpoint.lstrip('/')}"
            if method.upper() == "GET":
                future = self.session.get(full_url, timeout=timeout_value)
            else:
                future = self.session.post(full_url, json=payload, timeout=timeout_value)
            future_to_url[future] = url
        
        successful_urls = set()
        failed_details = {}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                response = future.result()
                if response.status_code == 200:
                    successful_urls.add(url)
                else:
                    failed_details[url] = f"HTTP {response.status_code}: {response.text[:100]}"
            except Exception as e:
                failed_details[url] = str(e)
        
        return successful_urls, failed_details
    
    def _wait_for_servers_ready(self, timeout: float):
        """Wait for all servers to become healthy within timeout"""
        start_time = time.time()
        
        while True:
            healthy_urls, failed_details = self._send_parallel_requests("health", "GET")
            if len(healthy_urls) == len(self._server_urls):
                return
            
            if time.time() - start_time > timeout:
                raise ConnectionError(f"Not all servers became ready within {timeout} seconds")
            
            time.sleep(1)
    
    def check_health(self) -> Dict[str, Any]:
        """Check health status of all servers"""
        healthy_urls, failed_details = self._send_parallel_requests("health", "GET")
        return {
            "healthy": sorted(healthy_urls),
            "unhealthy": sorted(failed_details.keys()),
            "errors": failed_details,
        }
    
    def _load_lora_adapter(self, lora_name: str, lora_path: Optional[str] = None) -> Dict[str, Any]:
        """Load LoRA adapter on all servers"""
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        successful_urls, failed_details = self._send_parallel_requests("v1/load_lora_adapter", "POST", payload)
        return {
            "success": len(failed_details) == 0,
            "successful_servers": sorted(successful_urls),
            "failed_servers": sorted(failed_details.keys()),
            "errors": failed_details
        }
    
    def _unload_lora_adapter(self, lora_name: str) -> Dict[str, Any]:
        """Unload LoRA adapter from all servers"""
        payload = {"lora_name": lora_name}
        successful_urls, failed_details = self._send_parallel_requests("v1/unload_lora_adapter", "POST", payload)
        return {
            "success": len(failed_details) == 0,
            "successful_servers": sorted(successful_urls),
            "failed_servers": sorted(failed_details.keys()),
            "errors": failed_details
        }
    
    def _reset_prefix_cache(self) -> Dict[str, Any]:
        """Reset prefix cache on all servers"""
        successful_urls, failed_details = self._send_parallel_requests("reset_prefix_cache", "POST")
        return {
            "success": len(failed_details) == 0,
            "successful_servers": sorted(successful_urls),
            "failed_servers": sorted(failed_details.keys()),
            "errors": failed_details
        }
    
    def _sleep(self, level: int = 1) -> Dict[str, Any]:
        """Set sleep state on all servers"""
        payload = {"level": level}
        successful_urls, failed_details = self._send_parallel_requests("sleep", "POST", payload)
        return {
            "success": len(failed_details) == 0,
            "sleeping_servers": sorted(successful_urls),
            "failed_servers": sorted(failed_details.keys()),
            "errors": failed_details
        }
    
    def _is_sleeping(self) -> Dict[str, Any]:
        """Check if servers are in sleep state"""
        sleeping_urls, failed_details = self._send_parallel_requests("is_sleeping", "GET")
        # For this endpoint, "failed" means awake, not actually failed
        awake_servers = list(failed_details.keys())
        return {
            "all_sleeping": len(awake_servers) == 0,
            "sleeping_servers": sorted(sleeping_urls),
            "awake_servers": sorted(awake_servers),
            "errors": {url: err for url, err in failed_details.items() if not err.startswith("HTTP 200")}
        }
    
    def _wake_up(self) -> Dict[str, Any]:
        """Wake up all servers"""
        successful_urls, failed_details = self._send_parallel_requests("wake_up", "POST")
        return {
            "success": len(failed_details) == 0,
            "awakened_servers": sorted(successful_urls),
            "failed_servers": sorted(failed_details.keys()),
            "errors": failed_details
        }
    
    @property
    def server_urls(self) -> list[str]:
        return self._server_urls


    def sleep(self):
        """Put inference servers to sleep to free GPU memory"""
        result = self._sleep()
        if not result["success"]:
            raise RuntimeError(f"Failed to put servers to sleep: {result['errors']}")
    
    def wake_up(self):
        """Wake up inference servers from sleep"""
        result = self._wake_up()
        if not result["success"]:
            raise RuntimeError(f"Failed to wake up servers: {result['errors']}")
    
    def load_weights_from_disk(self, weights_path: str | Path):
        """Load full model weights from disk (placeholder - not implemented yet)"""
        raise NotImplementedError("Full weight loading not yet implemented for VLLM engine")

    def load_lora_weights_from_disk(self, weights_path: str | Path, name: str = "default_lora") -> bool:
        wake_up_result = self._wake_up()
        if not wake_up_result["success"]:
            raise RuntimeError(f"Failed to wake up servers: {wake_up_result['errors']}")
        
        reset_prefix_cache_result = self._reset_prefix_cache()
        if not reset_prefix_cache_result["success"]:
            raise RuntimeError(f"Failed to reset prefix cache: {reset_prefix_cache_result['errors']}")
        
        unload_lora_result = self._unload_lora_adapter(name)
        
        load_lora_result = self._load_lora_adapter(name, str(weights_path))
        if not load_lora_result["success"]:
            raise RuntimeError(f"Failed to load LoRA adapter: {load_lora_result['errors']}")
        
        return True