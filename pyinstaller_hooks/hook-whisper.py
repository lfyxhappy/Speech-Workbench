from PyInstaller.utils.hooks import collect_data_files

module_collection_mode = {
    "whisper": "pyz+py",
}

datas = collect_data_files("whisper", includes=["assets/*", "normalizers/*"])
