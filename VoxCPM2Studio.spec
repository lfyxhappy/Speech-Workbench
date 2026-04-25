# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []
datas += collect_data_files('PyQt6.QtMultimedia')
datas += copy_metadata('accelerate')
datas += copy_metadata('av')
datas += copy_metadata('certifi')
datas += copy_metadata('charset-normalizer')
datas += copy_metadata('ctranslate2')
datas += copy_metadata('diffusers')
datas += copy_metadata('faster-whisper')
datas += copy_metadata('filelock')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('httpcore')
datas += copy_metadata('httpx')
datas += copy_metadata('idna')
datas += copy_metadata('importlib-metadata')
datas += copy_metadata('librosa')
datas += copy_metadata('numpy')
datas += copy_metadata('openai-whisper')
datas += copy_metadata('packaging')
datas += copy_metadata('Pillow')
datas += copy_metadata('protobuf')
datas += copy_metadata('PyYAML')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('safetensors')
datas += copy_metadata('scipy')
datas += copy_metadata('sentencepiece')
datas += copy_metadata('soundfile')
datas += copy_metadata('tokenizers')
datas += copy_metadata('torch')
datas += copy_metadata('torchcodec')
datas += copy_metadata('tqdm')
datas += copy_metadata('transformers')
datas += copy_metadata('urllib3')
datas += copy_metadata('zipp')
binaries += collect_dynamic_libs('PyQt6.QtMultimedia')


a = Analysis(
    ['C:\\算法\\小应用\\各种各样的模型\\语音生成\\voxcpm_gui.py'],
    pathex=['C:\\算法\\小应用\\各种各样的模型\\语音生成'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['PyQt6.QtMultimedia', 'whisper', 'faster_whisper', 'ctranslate2', 'tiktoken', 'diffusers', 'accelerate', 'sentencepiece', 'audiofx_service', 'diffusers.pipelines.audioldm2', 'diffusers.pipelines.audioldm2.pipeline_audioldm2'],
    hookspath=['C:\\算法\\小应用\\各种各样的模型\\语音生成\\pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['onnxruntime', 'onnxruntime-gpu', 'onnxruntime.training'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoxCPM2Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoxCPM2Studio',
)
