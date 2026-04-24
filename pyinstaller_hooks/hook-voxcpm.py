"""PyInstaller hook for VoxCPM.

TorchScript uses inspect/getsource when importing the AudioVAE modules.
The frozen app therefore needs the original .py sources on disk instead of
only bytecode in the PYZ archive.
"""

module_collection_mode = {
    "voxcpm": "pyz+py",
    "voxcpm.modules": "pyz+py",
    "voxcpm.modules.audiovae": "pyz+py",
    "voxcpm.modules.audiovae.audio_vae": "pyz+py",
    "voxcpm.modules.audiovae.audio_vae_v2": "pyz+py",
}
