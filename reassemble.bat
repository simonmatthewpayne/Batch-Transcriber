@echo off
echo ------------------------------------------
echo Reassembling WhisperBatchTranscriber.exe...
echo ------------------------------------------

copy /b WhisperBatchTranscriber_part1.bin + WhisperBatchTranscriber_part2.bin WhisperBatchTranscriber.exe >nul

if exist WhisperBatchTranscriber.exe (
    echo ✅ Done! File rebuilt successfully.
    echo WhisperBatchTranscriber.exe is now in this folder.
) else (
    echo ❌ Something went wrong. Make sure both .bin files are in the same folder.
)

pause
