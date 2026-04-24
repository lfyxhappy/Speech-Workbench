# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = []
binaries = []
datas += collect_data_files('PyQt6.QtMultimedia')
binaries += collect_dynamic_libs('PyQt6.QtMultimedia')


a = Analysis(
    ['C:\\算法\\小应用\\各种各样的模型\\语音生成\\voxcpm_gui.py'],
    pathex=['C:\\算法\\小应用\\各种各样的模型\\语音生成'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['PyQt6.QtMultimedia', 'whisper', 'faster_whisper', 'ctranslate2', 'tiktoken'],
    hookspath=['C:\\算法\\小应用\\各种各样的模型\\语音生成\\pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
