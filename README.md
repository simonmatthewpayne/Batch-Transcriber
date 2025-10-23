# 🎧 Whisper Batch Transcriber

**Version 1.1 — CUDA + DirectML Edition**  
Created by **Simon Matthew Payne (2025)**

A simple, one-click desktop app that automatically transcribes any audio or video file using [OpenAI Whisper](https://github.com/openai/whisper).  
Runs **entirely offline**, supports **NVIDIA (CUDA)**, **AMD & Intel (DirectML)**, or **CPU**, and saves plain-text transcripts next to your originals.  

---

## 🚀 Quick Start

# Whisper Batch Transcriber

Offline-ready batch audio-to-text transcriber built with [OpenAI Whisper](https://github.com/openai/whisper).

---

## 🔧 Rebuilding the App

1. **Download both files**:
   - `WhisperBatchTranscriber_part1.bin` https://github.com/simonmatthewpayne/Batch-Transcriber/releases/download/untagged-7600cffb1c97574dcf58/WhisperBatchTranscriber_part1.bin
   - `WhisperBatchTranscriber_part2.bin` https://github.com/simonmatthewpayne/Batch-Transcriber/releases/download/untagged-7600cffb1c97574dcf58/WhisperBatchTranscriber_part2.bin

2. **Place them in the same folder** as `reassemble.bat` https://github.com/simonmatthewpayne/Batch-Transcriber/releases/download/untagged-7600cffb1c97574dcf58/reassemble.bat.

3. **Double-click `reassemble.bat`** — it will rebuild:

3. **Double-click `WhisperBatchTranscriber.exe`**  
4. Click **Select files …**, choose one or many audio/video files.  
   - Supported: `mp3`, `wav`, `m4a`, `flac`, `aac`, `ogg`, `opus`, `wma`, `mp4`, `mkv`, `mov`, `avi`, `webm`, and more.  
5. Leave **Model** and **Device** on “Auto” (recommended).  
   - The app detects your GPU and chooses a sensible model size automatically.  
6. Click **Run** to start transcribing.  
7. When finished, you’ll find a `*_transcript.txt` next to each original file.

---

## 🧠 Features

| Capability | Description |
|-------------|-------------|
| 🎞️ **Batch processing** | Transcribe many files at once. |
| ⚙️ **Auto hardware detection** | CUDA → DirectML → CPU fallback chain. |
| 💡 **Model recommendations** | Chooses `tiny / base / small / medium / large` based on VRAM and RAM. |
| 🔄 **Cancel controls** | Cancel selected pending files or stop after current file safely. |
| 📊 **Progress bar + status** | See which file is running and how many are done. |
| 🧩 **Bundled FFmpeg** | Works out of the box — no extra downloads. |
| 💾 **Offline & private** | All transcription happens locally on your machine. |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|------------|----------|-------------|
| OS | Windows 10 / 11 (64-bit) | Latest Windows 11 |
| RAM | 8 GB | 16 GB + |
| GPU | Optional | NVIDIA GPU (CUDA) / AMD or Intel GPU (DirectML) |

### GPU acceleration details

| Vendor | Backend | Support |
|---------|----------|----------|
| **NVIDIA (GeForce / RTX)** | CUDA (Pytorch) | ✅ Full speed |
| **AMD (Radeon)** | DirectML (Windows only) | ⚙️ Moderate speed |
| **Intel (iGPU / Arc)** | DirectML (Windows only) | ⚙️ Moderate speed |
| **No GPU** | CPU | 🐢 Works (but slower) |

> The app automatically picks the best available backend.  
> No manual driver configuration required.

---

## 🧩 Troubleshooting

**“ffmpeg not found”**  
> Shouldn’t happen — ffmpeg is bundled. If it does, ensure `ffmpeg/ffmpeg.exe` exists next to the EXE.

**“CUDA not available”**  
> You don’t have an NVIDIA GPU or drivers installed.  
> The app will use DirectML or CPU automatically.

**“Slow performance”**  
> Try a smaller model (`base` or `tiny`) or ensure GPU acceleration is enabled (auto by default).

---

## ⚖️ Licences & Acknowledgements

- 🧠 **Whisper** — © OpenAI (MIT License)  
- 🔥 **PyTorch** — © Meta AI (BSD License)  
- 🎬 **FFmpeg** — LGPL/GPL License (https://ffmpeg.org/)  
  - See `FFMPEG_LICENSE.txt` for licence text.  
- 🪄 **torch-directml** — © Microsoft (DirectML backend for PyTorch)

---

## 💬 Credits

Developed by **Simon Matthew Payne**, PhD in Condensed Matter Physics  
University of Bristol  · 2025  

> _“Easy-mode transcription for everyone — just drag, drop, and go.”_
