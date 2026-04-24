from PyInstaller.utils.hooks import collect_data_files

module_collection_mode = {
    "whisper": "pyz+py",
    "faster_whisper": "pyz+py",
    "ctranslate2": "pyz+py",
    "tiktoken": "pyz+py",
}

datas = []
datas += collect_data_files("whisper", includes=["assets/*", "normalizers/*"])
