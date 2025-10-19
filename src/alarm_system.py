"""
Alarm System Module
Handles audio and visual alarms for safety violations
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
import winsound

class AlarmSystem:
    """Alarm system for safety violations"""
    
    def __init__(self):
        """Initialize the alarm system"""
        self.logger = logging.getLogger(__name__)
        self.alarm_active = False
        self.alarm_thread = None
        
        # Initialize audio system (using winsound for Windows)
        try:
            # Test if winsound is available
            winsound.Beep(800, 100)
            self.logger.info("✅ Audio system initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize audio: {str(e)}")
    
    def trigger_alarm(self, violations):
        """
        Trigger alarm for safety violations
        
        Args:
            violations (list): List of violation messages
        """
        if not violations:
            return
        
        self.logger.warning(f"🚨 Safety violations detected: {violations}")
        
        # Log violations to file
        self._log_violations(violations)
        
        # Start alarm in separate thread
        if not self.alarm_active:
            self.alarm_thread = threading.Thread(
                target=self._run_alarm_sequence,
                args=(violations,),
                daemon=True
            )
            self.alarm_thread.start()
    
    def _run_alarm_sequence(self, violations):
        """Run alarm sequence for violations"""
        self.alarm_active = True
        
        try:
            # Play alarm sound
            self._play_alarm_sound()
            
            # Show visual alert (this will be handled by the UI)
            self._show_visual_alert(violations)
            
            # Wait for alarm duration
            time.sleep(5)  # 5 second alarm
            
        except Exception as e:
            self.logger.error(f"❌ Error in alarm sequence: {str(e)}")
        finally:
            self.alarm_active = False
    
    def _play_alarm_sound(self):
        """Play alarm sound"""
        try:
            # Play alarm sound using winsound (Windows built-in)
            # Play beep sequence: high frequency beeps
            for _ in range(3):
                winsound.Beep(1000, 300)  # High frequency beep
                time.sleep(0.2)
                winsound.Beep(800, 300)   # Lower frequency beep
                time.sleep(0.2)
                
        except Exception as e:
            self.logger.error(f"❌ Error playing alarm sound: {str(e)}")
            # Fallback to simple beep
            try:
                winsound.Beep(800, 500)
            except:
                pass
    
    def _show_visual_alert(self, violations):
        """Show visual alert (handled by UI)"""
        # This method is called to indicate visual alert should be shown
        # The actual visual alert is handled by the Streamlit UI
        pass
    
    def _log_violations(self, violations):
        """Log violations to file"""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Log file path
            log_file = logs_dir / "safety_violations.log"
            
            # Write violation log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] Safety Violations Detected:\n")
                for violation in violations:
                    f.write(f"  - {violation}\n")
                f.write("-" * 50 + "\n")
            
            self.logger.info(f"📝 Violations logged to {log_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging violations: {str(e)}")
    
    def stop_alarm(self):
        """Stop the alarm"""
        self.alarm_active = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1)
    
    def get_alarm_status(self):
        """Get current alarm status"""
        return self.alarm_active
    
    def get_violation_history(self, limit=50):
        """
        Get recent violation history
        
        Args:
            limit (int): Maximum number of violations to return
            
        Returns:
            list: List of recent violations
        """
        try:
            log_file = Path("logs") / "safety_violations.log"
            
            if not log_file.exists():
                return []
            
            violations = []
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Parse violation entries
            current_entry = None
            for line in lines:
                line = line.strip()
                if line.startswith("[") and "Safety Violations Detected:" in line:
                    # Extract timestamp
                    timestamp = line.split("]")[0][1:]
                    current_entry = {"timestamp": timestamp, "violations": []}
                elif line.startswith("- ") and current_entry:
                    # Extract violation
                    violation = line[2:]
                    current_entry["violations"].append(violation)
                elif line.startswith("-" * 50) and current_entry:
                    # End of entry
                    violations.append(current_entry)
                    current_entry = None
            
            # Return most recent violations
            return violations[-limit:] if violations else []
            
        except Exception as e:
            self.logger.error(f"❌ Error reading violation history: {str(e)}")
            return []
