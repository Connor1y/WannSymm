"""
Useful I/O utilities for WannSymm

System resource monitoring and I/O utilities.
Translated from: src/usefulio.h and src/usefulio.c

Translation Status: ✅ COMPLETED
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring will be limited")


# Configure logging
logger = logging.getLogger(__name__)


def get_memory_usage() -> int:
    """
    Get current process memory usage in bytes.
    
    Returns the resident set size (RSS) - the portion of memory occupied 
    by the process that is held in RAM.
    
    Returns:
        int: Memory usage in bytes. Returns -1 if psutil is not available
             or an error occurs, 0 if platform is unsupported.
    
    Examples:
        >>> mem_bytes = get_memory_usage()
        >>> if mem_bytes > 0:
        ...     print(f"Using {mem_bytes / (1024**3):.3f} GiB")
    
    Note:
        This function uses psutil for cross-platform memory monitoring.
        If psutil is not available, it returns -1.
    """
    if not PSUTIL_AVAILABLE:
        logger.debug("psutil not available for memory monitoring")
        return -1
    
    try:
        process = psutil.Process()
        # Get resident set size (RSS) in bytes
        mem_info = process.memory_info()
        return mem_info.rss
    except (psutil.Error, OSError) as e:
        logger.error(f"Failed to get memory usage: {e}")
        return -1


def get_memory_usage_str(prefix: str = "") -> str:
    """
    Get formatted memory usage string with appropriate units.
    
    Formats memory usage as GiB, MiB, or KiB depending on magnitude.
    The format matches the C version for compatibility.
    
    Args:
        prefix: Optional prefix string to prepend to the output.
                Default is empty string.
    
    Returns:
        str: Formatted string like "prefix123.456 GiB" where units are
             chosen based on memory size (GiB for > 1 GiB, MiB for > 1 MiB,
             else KiB).
    
    Examples:
        >>> mem_str = get_memory_usage_str("Memory: ")
        >>> print(mem_str)  # e.g., "Memory: 2.345 GiB"
        
        >>> mem_str = get_memory_usage_str()
        >>> print(mem_str)  # e.g., "512.123 MiB"
    
    Note:
        Returns a string indicating unavailable memory monitoring if
        get_memory_usage() returns an error value.
    """
    mem_usage_bytes = get_memory_usage()
    
    if mem_usage_bytes < 0:
        return f"{prefix}N/A (memory monitoring unavailable)"
    
    # Format with appropriate units (GiB, MiB, or KiB)
    if mem_usage_bytes > 1024 * 1024 * 1024:
        # GiB
        mem_value = mem_usage_bytes / (1024 * 1024 * 1024)
        return f"{prefix}{mem_value:.3f} GiB"
    elif mem_usage_bytes > 1024 * 1024:
        # MiB
        mem_value = mem_usage_bytes / (1024 * 1024)
        return f"{prefix}{mem_value:.3f} MiB"
    else:
        # KiB
        mem_value = mem_usage_bytes / 1024
        return f"{prefix}{mem_value:.3f} KiB"


class ProgressFile:
    """
    Progress file writer for thread/process progress tracking.
    
    This class manages writing progress updates to files with names like
    ".progress-of-threadN" where N is the thread/rank number. This matches
    the behavior in the C code for MPI parallel execution.
    
    Attributes:
        thread_id: Thread or rank identifier (1-indexed).
        filename: Path to the progress file.
        base_dir: Directory where progress file is written.
    
    Examples:
        >>> pf = ProgressFile(thread_id=1)
        >>> pf.write_progress(
        ...     symm_no=1, 
        ...     progress=50.0, 
        ...     current=10, 
        ...     total=20
        ... )
        
        >>> pf.clear()  # Remove the progress file
    """
    
    def __init__(self, thread_id: int = 1, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize progress file writer.
        
        Args:
            thread_id: Thread or rank number (1-indexed, following MPI convention).
                       Default is 1.
            base_dir: Directory for progress file. Default is current directory.
        
        Raises:
            ValueError: If thread_id is less than 1.
        """
        if thread_id < 1:
            raise ValueError(f"thread_id must be >= 1, got {thread_id}")
        
        self.thread_id = thread_id
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.filename = self.base_dir / f".progress-of-thread{thread_id}"
        
        logger.debug(f"ProgressFile initialized for thread {thread_id}: {self.filename}")
    
    def clear(self) -> None:
        """
        Remove the progress file if it exists.
        
        This is typically called at the start of processing to clear
        any old progress information.
        
        Examples:
            >>> pf = ProgressFile(thread_id=1)
            >>> pf.clear()
        """
        try:
            if self.filename.exists():
                self.filename.unlink()
                logger.debug(f"Cleared progress file: {self.filename}")
        except OSError as e:
            logger.warning(f"Failed to clear progress file {self.filename}: {e}")
    
    def write_progress(
        self,
        symm_no: int,
        progress: float,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> None:
        """
        Write progress information to the file.
        
        Appends a progress line to the file. The format matches the C version:
        "Symm No. {symm_no}, progress {progress:.2f}% ({current}/{total})"
        
        Args:
            symm_no: Symmetry operation number (1-indexed).
            progress: Progress percentage (0-100).
            current: Current iteration/item number.
            total: Total number of iterations/items.
            message: Optional additional message to append.
        
        Examples:
            >>> pf = ProgressFile(thread_id=1)
            >>> pf.write_progress(symm_no=1, progress=25.5, current=10, total=40)
            # Writes: "Symm No. 1, progress 25.50% (10/40)"
            
            >>> pf.write_progress(
            ...     symm_no=2, 
            ...     progress=50.0, 
            ...     current=20, 
            ...     total=40,
            ...     message="Processing k-points"
            ... )
            # Writes: "Symm No. 2, progress 50.00% (20/40) - Processing k-points"
        
        Note:
            Uses append mode ('a') so multiple progress updates accumulate
            in the file.
        """
        try:
            # Format the progress message (matching C format)
            progress_msg = (
                f"Symm No. {symm_no}, progress {progress:5.2f}% "
                f"({current}/{total})"
            )
            
            if message:
                progress_msg += f" - {message}"
            
            # Append to file (creating if needed)
            with self.filename.open('a') as f:
                f.write(progress_msg + '\n')
            
            logger.debug(f"Progress written: {progress_msg}")
            
        except OSError as e:
            logger.error(f"Failed to write progress to {self.filename}: {e}")
    
    def read_progress(self) -> list:
        """
        Read all progress lines from the file.
        
        Returns:
            list: List of progress lines, or empty list if file doesn't exist
                  or cannot be read.
        
        Examples:
            >>> pf = ProgressFile(thread_id=1)
            >>> pf.write_progress(symm_no=1, progress=50.0, current=5, total=10)
            >>> lines = pf.read_progress()
            >>> print(lines[0])
            'Symm No. 1, progress 50.00% (5/10)'
        """
        try:
            if not self.filename.exists():
                return []
            
            with self.filename.open('r') as f:
                return [line.rstrip('\n') for line in f]
        
        except OSError as e:
            logger.error(f"Failed to read progress from {self.filename}: {e}")
            return []


class Timer:
    """
    Simple timer for measuring elapsed time.
    
    This class provides convenient timing utilities for both wall-clock
    (real) time and CPU time measurements.
    
    Attributes:
        start_time: Wall-clock start time (from time.time()).
        start_cpu: CPU time at start (from time.process_time()).
    
    Examples:
        >>> timer = Timer()
        >>> # ... do some work ...
        >>> real_time = timer.elapsed()
        >>> cpu_time = timer.cpu_time()
        >>> print(f"Real: {real_time:.2f}s, CPU: {cpu_time:.2f}s")
        
        >>> timer.reset()  # Start timing again
    """
    
    def __init__(self):
        """Initialize timer with current time."""
        self.start_time = time.time()
        self.start_cpu = time.process_time()
        logger.debug("Timer started")
    
    def reset(self) -> None:
        """
        Reset the timer to current time.
        
        Examples:
            >>> timer = Timer()
            >>> # ... do some work ...
            >>> timer.reset()  # Start over
        """
        self.start_time = time.time()
        self.start_cpu = time.process_time()
        logger.debug("Timer reset")
    
    def elapsed(self) -> float:
        """
        Get elapsed wall-clock time in seconds.
        
        Returns:
            float: Seconds elapsed since timer start or last reset.
        
        Examples:
            >>> timer = Timer()
            >>> time.sleep(1.0)
            >>> elapsed = timer.elapsed()
            >>> assert elapsed >= 1.0
        """
        return time.time() - self.start_time
    
    def cpu_time(self) -> float:
        """
        Get elapsed CPU time in seconds.
        
        CPU time only counts time when the process is actively using the CPU,
        not time spent sleeping or waiting for I/O.
        
        Returns:
            float: CPU seconds used since timer start or last reset.
        
        Examples:
            >>> timer = Timer()
            >>> # ... do CPU-intensive work ...
            >>> cpu = timer.cpu_time()
            >>> print(f"CPU time: {cpu:.2f}s")
        """
        return time.process_time() - self.start_cpu
    
    def format_elapsed(self, include_cpu: bool = False) -> str:
        """
        Get formatted elapsed time string.
        
        Args:
            include_cpu: If True, include CPU time in addition to real time.
        
        Returns:
            str: Formatted time string like "real time elapsed 123 s" or
                 "real time elapsed 123 s, cpu time 45.67 s" if include_cpu is True.
        
        Examples:
            >>> timer = Timer()
            >>> time.sleep(1.0)
            >>> print(timer.format_elapsed())
            'real time elapsed 1 s'
            
            >>> print(timer.format_elapsed(include_cpu=True))
            'real time elapsed 1 s, cpu time 0.01 s'
        """
        elapsed = self.elapsed()
        msg = f"real time elapsed {elapsed:.0f} s"
        
        if include_cpu:
            cpu = self.cpu_time()
            msg += f", cpu time {cpu:.2f} s"
        
        return msg
