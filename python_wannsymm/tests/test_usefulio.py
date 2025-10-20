"""
Tests for usefulio module

Test memory monitoring, timing utilities, and progress file I/O.
"""

import pytest
import time
import tempfile
from pathlib import Path

from wannsymm.usefulio import (
    get_memory_usage,
    get_memory_usage_str,
    ProgressFile,
    Timer,
    PSUTIL_AVAILABLE
)


class TestMemoryUsage:
    """Test memory usage monitoring functions."""
    
    def test_get_memory_usage_returns_int(self):
        """Test that get_memory_usage returns an integer."""
        mem_usage = get_memory_usage()
        assert isinstance(mem_usage, int)
    
    def test_get_memory_usage_value(self):
        """Test that memory usage is reasonable (>0 or -1 if unavailable)."""
        mem_usage = get_memory_usage()
        if PSUTIL_AVAILABLE:
            assert mem_usage > 0, "Memory usage should be positive when psutil is available"
        else:
            assert mem_usage == -1, "Memory usage should be -1 when psutil is unavailable"
    
    def test_get_memory_usage_str_format(self):
        """Test that get_memory_usage_str returns a properly formatted string."""
        mem_str = get_memory_usage_str()
        assert isinstance(mem_str, str)
        
        if PSUTIL_AVAILABLE:
            # Should contain a unit (GiB, MiB, or KiB)
            assert any(unit in mem_str for unit in ['GiB', 'MiB', 'KiB'])
        else:
            # Should indicate unavailable
            assert 'N/A' in mem_str or 'unavailable' in mem_str.lower()
    
    def test_get_memory_usage_str_with_prefix(self):
        """Test get_memory_usage_str with custom prefix."""
        prefix = "Memory: "
        mem_str = get_memory_usage_str(prefix)
        assert mem_str.startswith(prefix)
    
    def test_get_memory_usage_str_units(self):
        """Test that get_memory_usage_str uses appropriate units."""
        mem_str = get_memory_usage_str()
        
        if PSUTIL_AVAILABLE:
            # Verify format is "X.XXX Unit"
            parts = mem_str.split()
            if len(parts) >= 2:
                # Should have a number and a unit
                try:
                    value = float(parts[-2])
                    assert value >= 0
                    assert parts[-1] in ['GiB', 'MiB', 'KiB']
                except (ValueError, IndexError):
                    # If format is different, at least check for unit presence
                    assert any(unit in mem_str for unit in ['GiB', 'MiB', 'KiB'])
    
    def test_memory_usage_consistency(self):
        """Test that repeated calls give consistent results."""
        mem1 = get_memory_usage()
        mem2 = get_memory_usage()
        
        if PSUTIL_AVAILABLE:
            # Memory should be in the same ballpark (within 10 MB typically)
            assert abs(mem1 - mem2) < 10 * 1024 * 1024, \
                "Memory usage should be relatively stable between consecutive calls"


class TestProgressFile:
    """Test ProgressFile class for progress tracking."""
    
    def test_progress_file_creation(self):
        """Test creating a ProgressFile instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            assert pf.thread_id == 1
            assert pf.filename.name == ".progress-of-thread1"
            assert pf.base_dir == Path(tmpdir)
    
    def test_progress_file_default_directory(self):
        """Test ProgressFile with default directory."""
        pf = ProgressFile(thread_id=2)
        assert pf.thread_id == 2
        assert pf.filename.name == ".progress-of-thread2"
    
    def test_progress_file_invalid_thread_id(self):
        """Test that invalid thread_id raises ValueError."""
        with pytest.raises(ValueError):
            ProgressFile(thread_id=0)
        
        with pytest.raises(ValueError):
            ProgressFile(thread_id=-1)
    
    def test_progress_file_write(self):
        """Test writing progress to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            pf.write_progress(symm_no=1, progress=50.0, current=5, total=10)
            
            # Verify file exists
            assert pf.filename.exists()
            
            # Read and verify content
            with pf.filename.open('r') as f:
                content = f.read()
            
            assert "Symm No. 1" in content
            assert "50.00%" in content
            assert "(5/10)" in content
    
    def test_progress_file_write_with_message(self):
        """Test writing progress with optional message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            pf.write_progress(
                symm_no=2,
                progress=75.5,
                current=15,
                total=20,
                message="Processing k-points"
            )
            
            with pf.filename.open('r') as f:
                content = f.read()
            
            assert "Symm No. 2" in content
            assert "75.50%" in content
            assert "(15/20)" in content
            assert "Processing k-points" in content
    
    def test_progress_file_append_mode(self):
        """Test that multiple writes append to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            # Write multiple progress updates
            pf.write_progress(symm_no=1, progress=25.0, current=1, total=4)
            pf.write_progress(symm_no=1, progress=50.0, current=2, total=4)
            pf.write_progress(symm_no=1, progress=75.0, current=3, total=4)
            
            # Read all lines
            lines = pf.read_progress()
            
            assert len(lines) == 3
            assert "25.00%" in lines[0]
            assert "50.00%" in lines[1]
            assert "75.00%" in lines[2]
    
    def test_progress_file_clear(self):
        """Test clearing progress file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            # Write some content
            pf.write_progress(symm_no=1, progress=50.0, current=5, total=10)
            assert pf.filename.exists()
            
            # Clear the file
            pf.clear()
            assert not pf.filename.exists()
    
    def test_progress_file_clear_nonexistent(self):
        """Test clearing non-existent file doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            # Should not raise error even if file doesn't exist
            pf.clear()
    
    def test_progress_file_read_empty(self):
        """Test reading from non-existent progress file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            lines = pf.read_progress()
            assert lines == []
    
    def test_progress_file_read_after_write(self):
        """Test reading progress after writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            pf.write_progress(symm_no=1, progress=33.33, current=1, total=3)
            pf.write_progress(symm_no=1, progress=66.67, current=2, total=3)
            
            lines = pf.read_progress()
            assert len(lines) == 2
            assert "33.33%" in lines[0]
            assert "66.67%" in lines[1]
    
    def test_progress_file_multiple_threads(self):
        """Test multiple ProgressFile instances for different threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf1 = ProgressFile(thread_id=1, base_dir=tmpdir)
            pf2 = ProgressFile(thread_id=2, base_dir=tmpdir)
            
            pf1.write_progress(symm_no=1, progress=50.0, current=5, total=10)
            pf2.write_progress(symm_no=2, progress=75.0, current=15, total=20)
            
            # Verify both files exist independently
            assert pf1.filename.exists()
            assert pf2.filename.exists()
            assert pf1.filename != pf2.filename
            
            # Verify content is separate
            lines1 = pf1.read_progress()
            lines2 = pf2.read_progress()
            
            assert len(lines1) == 1
            assert len(lines2) == 1
            assert "Symm No. 1" in lines1[0]
            assert "Symm No. 2" in lines2[0]


class TestTimer:
    """Test Timer class for time measurement."""
    
    def test_timer_creation(self):
        """Test creating a Timer instance."""
        timer = Timer()
        assert hasattr(timer, 'start_time')
        assert hasattr(timer, 'start_cpu')
    
    def test_timer_elapsed_increases(self):
        """Test that elapsed time increases."""
        timer = Timer()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = timer.elapsed()
        
        assert elapsed >= 0.1, "Elapsed time should be at least 0.1 seconds"
        assert elapsed < 1.0, "Elapsed time should be less than 1 second"
    
    def test_timer_cpu_time(self):
        """Test CPU time measurement."""
        timer = Timer()
        
        # Do some CPU work
        _ = sum(i * i for i in range(10000))
        
        cpu_time = timer.cpu_time()
        assert cpu_time >= 0, "CPU time should be non-negative"
    
    def test_timer_reset(self):
        """Test resetting the timer."""
        timer = Timer()
        time.sleep(0.1)
        
        elapsed_before = timer.elapsed()
        assert elapsed_before >= 0.1
        
        # Reset and check again
        timer.reset()
        elapsed_after = timer.elapsed()
        
        assert elapsed_after < elapsed_before, \
            "Elapsed time after reset should be less than before"
        assert elapsed_after < 0.01, \
            "Elapsed time immediately after reset should be very small"
    
    def test_timer_format_elapsed_basic(self):
        """Test formatted elapsed time string."""
        timer = Timer()
        time.sleep(0.1)
        
        formatted = timer.format_elapsed()
        assert isinstance(formatted, str)
        assert "real time elapsed" in formatted
        assert "s" in formatted
    
    def test_timer_format_elapsed_with_cpu(self):
        """Test formatted elapsed time with CPU time."""
        timer = Timer()
        time.sleep(0.1)
        
        formatted = timer.format_elapsed(include_cpu=True)
        assert isinstance(formatted, str)
        assert "real time elapsed" in formatted
        assert "cpu time" in formatted
    
    def test_timer_multiple_measurements(self):
        """Test taking multiple time measurements."""
        timer = Timer()
        
        time.sleep(0.1)
        elapsed1 = timer.elapsed()
        
        time.sleep(0.1)
        elapsed2 = timer.elapsed()
        
        # Second measurement should be larger
        assert elapsed2 > elapsed1
        assert elapsed2 - elapsed1 >= 0.1
    
    def test_timer_cpu_vs_real_time(self):
        """Test that CPU time and real time can differ."""
        timer = Timer()
        
        # Sleep doesn't use CPU time
        time.sleep(0.1)
        
        real_time = timer.elapsed()
        cpu_time = timer.cpu_time()
        
        # Real time should be >= 0.1, CPU time should be much less
        assert real_time >= 0.1
        # CPU time during sleep should be minimal
        assert cpu_time < real_time


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_progress_tracking_simulation(self):
        """Simulate a typical progress tracking workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            timer = Timer()
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            # Clear any existing progress
            pf.clear()
            
            # Simulate processing loop
            total_items = 5
            for i in range(total_items):
                progress = (i + 1) / total_items * 100
                pf.write_progress(
                    symm_no=1,
                    progress=progress,
                    current=i + 1,
                    total=total_items,
                    message=f"Step {i+1}"
                )
                time.sleep(0.01)  # Simulate work
            
            # Verify progress was tracked
            lines = pf.read_progress()
            assert len(lines) == total_items
            
            # Verify timing
            elapsed = timer.elapsed()
            assert elapsed >= 0.05  # At least 5 * 0.01 seconds
    
    def test_memory_and_timing_together(self):
        """Test using memory monitoring and timing together."""
        timer = Timer()
        
        # Get initial memory
        mem1_str = get_memory_usage_str("Initial: ")
        
        # Do some work
        data = [i * i for i in range(10000)]
        
        # Get memory after work
        mem2_str = get_memory_usage_str("After work: ")
        
        elapsed = timer.elapsed()
        
        # Verify all measurements succeeded
        assert mem1_str
        assert mem2_str
        assert elapsed >= 0
    
    def test_realistic_workflow(self):
        """Test a realistic workflow with all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            timer = Timer()
            pf1 = ProgressFile(thread_id=1, base_dir=tmpdir)
            pf2 = ProgressFile(thread_id=2, base_dir=tmpdir)
            
            # Clear old progress
            pf1.clear()
            pf2.clear()
            
            # Simulate two parallel "threads"
            for symm_no in range(1, 3):
                pf = pf1 if symm_no == 1 else pf2
                
                for i in range(3):
                    progress = (i + 1) / 3 * 100
                    pf.write_progress(
                        symm_no=symm_no,
                        progress=progress,
                        current=i + 1,
                        total=3
                    )
            
            # Check results
            lines1 = pf1.read_progress()
            lines2 = pf2.read_progress()
            
            assert len(lines1) == 3
            assert len(lines2) == 3
            
            mem_str = get_memory_usage_str("Final memory: ")
            time_str = timer.format_elapsed(include_cpu=True)
            
            # All measurements should complete successfully
            assert mem_str
            assert time_str


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_progress_file_very_large_numbers(self):
        """Test progress file with very large numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            pf.write_progress(
                symm_no=999999,
                progress=99.999,
                current=999999,
                total=1000000
            )
            
            lines = pf.read_progress()
            assert len(lines) == 1
            assert "999999" in lines[0]
    
    def test_progress_file_zero_values(self):
        """Test progress file with zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = ProgressFile(thread_id=1, base_dir=tmpdir)
            
            pf.write_progress(
                symm_no=1,
                progress=0.0,
                current=0,
                total=100
            )
            
            lines = pf.read_progress()
            assert len(lines) == 1
            assert "0.00%" in lines[0]
    
    def test_timer_immediate_read(self):
        """Test reading timer immediately after creation."""
        timer = Timer()
        elapsed = timer.elapsed()
        cpu_time = timer.cpu_time()
        
        # Should be very close to zero but non-negative
        assert elapsed >= 0
        assert elapsed < 0.01
        assert cpu_time >= 0
    
    def test_multiple_timer_instances(self):
        """Test multiple independent timer instances."""
        timer1 = Timer()
        time.sleep(0.1)
        timer2 = Timer()
        time.sleep(0.1)
        
        elapsed1 = timer1.elapsed()
        elapsed2 = timer2.elapsed()
        
        # timer1 should have more elapsed time
        assert elapsed1 > elapsed2
        assert elapsed1 >= 0.2
        assert elapsed2 >= 0.1
        assert elapsed2 < 0.15
