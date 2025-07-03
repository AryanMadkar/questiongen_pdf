import threading
import time
import sys
import pickle
import gzip
from typing import Any, Optional

class AdvancedCache:
    def __init__(self, max_size=6000, ttl_seconds=14400, enable_compression=True):
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.cache_sizes = {}  # Track size of each cached item
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
        self.total_size = 0  # Track total cache size in bytes
        self.max_memory_mb = 500  # Maximum memory usage in MB
    
    def _get_object_size(self, obj: Any) -> int:
        """Get approximate size of object in bytes"""
        try:
            if self.enable_compression:
                # Get size of compressed object
                compressed_data = gzip.compress(pickle.dumps(obj))
                return len(compressed_data)
            else:
                return sys.getsizeof(pickle.dumps(obj))
        except Exception:
            # Fallback to basic size estimation
            return sys.getsizeof(str(obj))
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage"""
        try:
            pickled_data = pickle.dumps(data)
            return gzip.compress(pickled_data)
        except Exception as e:
            # Fallback to uncompressed storage
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage"""
        try:
            # Try to decompress first
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception:
            # If decompression fails, try direct unpickling
            try:
                return pickle.loads(compressed_data)
            except Exception as e:
                # If all fails, return None
                return None
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.creation_times[key] > self.ttl_seconds:
                    self._remove(key)
                    self.miss_count += 1
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                
                # Decompress and return data
                if self.enable_compression:
                    data = self._decompress_data(self.cache[key])
                    if data is not None:
                        self.hit_count += 1
                        return data
                    else:
                        # Corrupted data, remove from cache
                        self._remove(key)
                        self.miss_count += 1
                        return None
                else:
                    self.hit_count += 1
                    return self.cache[key]
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Any) -> bool:
        with self.lock:
            # Calculate size of new item
            if self.enable_compression:
                compressed_value = self._compress_data(value)
                item_size = len(compressed_value)
            else:
                item_size = self._get_object_size(value)
            
            # Check if item is too large (more than 50MB)
            if item_size > 50 * 1024 * 1024:
                return False
            
            # Remove existing item if key exists
            if key in self.cache:
                self._remove(key)
            
            # Ensure we have space
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            while (len(self.cache) >= self.max_size or 
                   self.total_size + item_size > max_memory_bytes) and self.cache:
                self._evict_lru()
            
            # Store the data
            if self.enable_compression:
                self.cache[key] = compressed_value
            else:
                self.cache[key] = value
            
            # Update metadata
            self.access_times[key] = time.time()
            self.creation_times[key] = time.time()
            self.cache_sizes[key] = item_size
            self.total_size += item_size
            
            return True
    
    def _remove(self, key: str):
        """Remove item from cache and update metadata"""
        if key in self.cache:
            # Update total size
            if key in self.cache_sizes:
                self.total_size -= self.cache_sizes[key]
                del self.cache_sizes[key]
            
            # Remove from all dictionaries
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.creation_times:
                del self.creation_times[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": round(self.total_size / (1024 * 1024), 2),
                "max_memory_mb": self.max_memory_mb,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds,
                "compression_enabled": self.enable_compression
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.cache_sizes.clear()
            self.total_size = 0
    
    def remove_expired(self):
        """Remove all expired entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, creation_time in self.creation_times.items():
                if current_time - creation_time > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def get_memory_usage(self) -> dict:
        """Get detailed memory usage information"""
        with self.lock:
            if not self.cache:
                return {"total_mb": 0, "average_item_mb": 0, "largest_item_mb": 0}
            
            total_mb = self.total_size / (1024 * 1024)
            average_item_mb = total_mb / len(self.cache)
            largest_item_mb = max(self.cache_sizes.values()) / (1024 * 1024) if self.cache_sizes else 0
            
            return {
                "total_mb": round(total_mb, 2),
                "average_item_mb": round(average_item_mb, 2),
                "largest_item_mb": round(largest_item_mb, 2),
                "compression_enabled": self.enable_compression
            }