@echo off
setlocal

:: ── llama-server launcher ────────────────────────────────────────────────────
:: Starts llama-server.exe with the CUDA backend and exposes an
:: OpenAI-compatible API at http://localhost:8080/v1
::
:: The Streamlit app should use:
::   DEEPAGENT_LLM_PROVIDER=llama_server
::   LLAMA_SERVER_BASE_URL=http://localhost:8080/v1

set MODEL=G:\llama_cpp\models\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
set PORT=8080
set N_CTX=8192
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
  if /I "%%A"=="LLAMA_SERVER_N_CTX" set "N_CTX=%%B"
)
if not "%LLAMA_SERVER_N_CTX%"=="" set N_CTX=%LLAMA_SERVER_N_CTX%
set N_GPU_LAYERS=-1
set HOST=127.0.0.1

:: Increase context for longer conversations (uses more VRAM).
:: Safe values on RTX 4090 24GB with 27B Q4: 8192 (safe) / 16384 (ok) / 32768 (tight)
:: set N_CTX=16384

echo.
echo  llama-server ^| model : %MODEL%
echo  llama-server ^| host  : http://%HOST%:%PORT%/v1
echo  llama-server ^| ctx   : %N_CTX%
echo  llama-server ^| gpu   : all layers (-ngl %N_GPU_LAYERS%)
echo.

G:\llama_cpp\llama-server.exe ^
  --model "%MODEL%" ^
  --ctx-size %N_CTX% ^
  --n-gpu-layers %N_GPU_LAYERS% ^
  --port %PORT% ^
  --host %HOST% ^
  --flash-attn on ^
  --log-disable

endlocal
