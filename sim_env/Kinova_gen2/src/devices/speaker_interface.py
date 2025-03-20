#!/usr/bin/env python3

import os
import time
import threading
import subprocess
import platform
from typing import Optional, Union

class SpeakerInterface:
    """Interface for text-to-speech and audio playback functionality."""
    
    def __init__(self, voice: str = "Alex", volume: float = 1.0, rate: float = 1.0, pitch: float = 1.0):
        """
        Initialize the speaker interface.
        
        Args:
            voice (str): Voice name (default: "Alex" for macOS, "en-US" for other platforms)
            volume (float): Volume level from 0.0 to 1.0
            rate (float): Speech rate multiplier (1.0 = normal speed)
            pitch (float): Voice pitch multiplier (1.0 = normal pitch)
        """
        self.voice = voice
        self._validate_and_set_audio_params(volume, rate, pitch)
        self._system = platform.system().lower()
        self._is_playing = False
        self._current_process = None
        self._check_dependencies()
        
        # Set default voice based on platform if needed
        if self._system == "darwin" and voice == "en-US":
            self.voice = "Alex"  # Default macOS voice
        elif self._system != "darwin" and voice == "Alex":
            self.voice = "en-US"  # Default voice for other platforms
    
    def _validate_and_set_audio_params(self, volume: float, rate: float, pitch: float):
        """Validate and set audio parameters."""
        # Validate and clamp volume to [0.0, 1.0]
        self.volume = max(0.0, min(volume, 1.0))
        
        # Validate and clamp rate to [0.5, 2.0]
        self.rate = max(0.5, min(rate, 2.0))
        
        # Validate and clamp pitch to [0.5, 2.0]
        self.pitch = max(0.5, min(pitch, 2.0))
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        if self._system == "linux":
            try:
                # Check for espeak
                subprocess.run(["espeak", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=False)
                self._tts_engine = "espeak"
            except (FileNotFoundError, subprocess.SubprocessError):
                try:
                    # Check for pico2wave
                    subprocess.run(["pico2wave", "--version"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   check=False)
                    self._tts_engine = "pico2wave"
                except (FileNotFoundError, subprocess.SubprocessError):
                    print("Warning: No supported TTS engine found. Install espeak or pico2wave.")
                    self._tts_engine = None
            
            # Check for aplay
            try:
                subprocess.run(["aplay", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=False)
                self._player = "aplay"
            except (FileNotFoundError, subprocess.SubprocessError):
                try:
                    # Check for mplayer
                    subprocess.run(["mplayer", "-v"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   check=False)
                    self._player = "mplayer"
                except (FileNotFoundError, subprocess.SubprocessError):
                    print("Warning: No supported audio player found. Install aplay or mplayer.")
                    self._player = None
        
        elif self._system == "darwin":  # macOS
            self._tts_engine = "say"
            self._player = "afplay"
        
        elif self._system == "windows":
            # Windows uses built-in speech API via powershell
            self._tts_engine = "powershell"
            self._player = "powershell"
        
        print(f"Speaker initialized with TTS engine: {self._tts_engine}, Audio player: {self._player}")
    
    def speak(self, text: str, wait: bool = True) -> bool:
        """
        Convert text to speech.
        
        Args:
            text (str): The text to speak
            wait (bool): Whether to wait for speech to complete before returning
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text.strip():
            print("Warning: Empty text provided, nothing to speak")
            return False
        
        if not self._tts_engine:
            print("Error: No text-to-speech engine available")
            return False
        
        # Clean up any current process
        self._cleanup_current_process()
        
        try:
            if self._system == "linux":
                if self._tts_engine == "espeak":
                    # Use espeak for TTS
                    cmd = [
                        "espeak",
                        "-v", self.voice,
                        "-a", str(int(self.volume * 200)),  # Volume (0-200)
                        "-s", str(int(self.rate * 175)),    # Speed (words per minute)
                        "-p", str(int(self.pitch * 50)),    # Pitch (0-99)
                        text
                    ]
                elif self._tts_engine == "pico2wave":
                    # Use pico2wave for TTS
                    temp_wav = "/tmp/pico_speech.wav"
                    cmd_tts = [
                        "pico2wave",
                        "-l", self.voice,
                        "-w", temp_wav,
                        text
                    ]
                    
                    # First generate the speech
                    subprocess.run(cmd_tts, check=True)
                    
                    # Then play it with the appropriate player
                    if self._player == "aplay":
                        cmd = ["aplay", temp_wav]
                    else:
                        cmd = ["mplayer", temp_wav]
            
            elif self._system == "darwin":  # macOS
                cmd = [
                    "say",
                    "-v", self.voice,
                    "-r", str(int(self.rate * 200))  # Rate (words per minute)
                ]
                
                # Add volume if available in newer macOS versions
                try:
                    subprocess.run(["say", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    output = subprocess.run(["say", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout.decode()
                    if "-v" in output:
                        cmd.extend(["-v", str(int(self.volume * 100))])
                except:
                    pass
                    
                cmd.append(text)  # Add the text to speak
            
            elif self._system == "windows":
                # Windows PowerShell command
                ps_script = (
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$speak.Volume = {int(self.volume * 100)}; '
                    f'$speak.Rate = {int((self.rate - 1.0) * 10)}; '
                    f'$speak.Speak([Console]::In.ReadToEnd())'
                )
                cmd = ["powershell", "-command", ps_script]
            
            # Execute the command
            print(f"Speaking: {text}")
            self._is_playing = True
            
            if wait:
                # Run synchronously and wait for completion
                self._current_process = subprocess.run(
                    cmd, 
                    text=(self._system == "windows"),  # Only use text mode for Windows
                    input=text if self._system == "windows" else None,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                self._is_playing = False
                self._current_process = None
                return True
            else:
                # Run asynchronously in a separate thread
                def _run_async():
                    try:
                        self._current_process = subprocess.Popen(
                            cmd,
                            text=(self._system == "windows"),
                            input=text if self._system == "windows" else None,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        self._current_process.wait()
                    except Exception as e:
                        print(f"Error in async speech: {e}")
                    finally:
                        self._is_playing = False
                        self._current_process = None
                
                threading.Thread(target=_run_async, daemon=True).start()
                return True
                
        except Exception as e:
            print(f"Error speaking text: {e}")
            self._is_playing = False
            self._current_process = None
            return False
    
    def play_audio(self, audio_file: str, wait: bool = True) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_file (str): Path to the audio file
            wait (bool): Whether to wait for playback to complete before returning
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return False
        
        if not self._player:
            print("Error: No audio player available")
            return False
        
        # Clean up any current process
        self._cleanup_current_process()
        
        try:
            if self._system == "linux":
                if self._player == "aplay":
                    cmd = ["aplay", audio_file]
                else:  # mplayer
                    cmd = ["mplayer", audio_file]
            
            elif self._system == "darwin":
                cmd = ["afplay", audio_file]
            
            elif self._system == "windows":
                # PowerShell command to play audio
                ps_script = (
                    f'(New-Object Media.SoundPlayer "{audio_file}").PlaySync();'
                )
                cmd = ["powershell", "-command", ps_script]
            
            # Execute the command
            print(f"Playing audio: {audio_file}")
            self._is_playing = True
            
            if wait:
                # Run synchronously and wait for completion
                self._current_process = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                self._is_playing = False
                self._current_process = None
                return True
            else:
                # Run asynchronously in a separate thread
                def _run_async():
                    try:
                        self._current_process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        self._current_process.wait()
                    except Exception as e:
                        print(f"Error in async audio playback: {e}")
                    finally:
                        self._is_playing = False
                        self._current_process = None
                
                threading.Thread(target=_run_async, daemon=True).start()
                return True
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            self._is_playing = False
            self._current_process = None
            return False
    
    def stop(self) -> bool:
        """
        Stop any currently playing speech or audio.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        return self._cleanup_current_process()
    
    def _cleanup_current_process(self) -> bool:
        """
        Clean up any currently running process.
        
        Returns:
            bool: True if cleanup was successful or no process was running
        """
        if self._current_process:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=1)
                self._current_process = None
                self._is_playing = False
                return True
            except Exception as e:
                print(f"Error stopping current process: {e}")
                return False
        return True
    
    def is_playing(self) -> bool:
        """
        Check if speech or audio is currently playing.
        
        Returns:
            bool: True if audio is currently playing
        """
        return self._is_playing
    
    def set_voice(self, voice: str) -> None:
        """Set the TTS voice."""
        self.voice = voice
    
    def set_volume(self, volume: float) -> None:
        """Set the volume level (0.0 to 1.0)."""
        self.volume = max(0.0, min(volume, 1.0))
    
    def set_rate(self, rate: float) -> None:
        """Set the speech rate multiplier (0.5 to 2.0)."""
        self.rate = max(0.5, min(rate, 2.0))
    
    def set_pitch(self, pitch: float) -> None:
        """Set the voice pitch multiplier (0.5 to 2.0)."""
        self.pitch = max(0.5, min(pitch, 2.0))

def main():
    """Test the speaker interface."""
    speaker = SpeakerInterface()
    
    print("\nTesting text-to-speech...")
    speaker.speak("Hello, I am a speaker interface for the AI agent. I can speak text and play audio files.")
    
    # Wait a moment
    time.sleep(1)
    
    print("\nChanging voice parameters...")
    speaker.set_rate(1.5)
    speaker.set_pitch(0.8)
    speaker.speak("Now I'm speaking faster with a lower pitch.")
    
    # Reset parameters
    speaker.set_rate(1.0)
    speaker.set_pitch(1.0)
    
    # You can uncomment to test audio file playback
    # print("\nTesting audio file playback...")
    # speaker.play_audio("path/to/your/audio/file.wav")
    
    print("\nTesting asynchronous speech...")
    speaker.speak("This is an asynchronous speech test. I will continue speaking while the program continues execution.", wait=False)
    
    # Do something else while speaking
    for i in range(5):
        print(f"Program is still running... {i+1}")
        time.sleep(0.5)
    
    # Wait for speech to complete
    while speaker.is_playing():
        time.sleep(0.1)
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 