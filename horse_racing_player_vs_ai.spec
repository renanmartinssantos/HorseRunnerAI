# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['horse_racing_player_vs_ai.py'],
    pathex=[],
    binaries=[],
    datas=[('images', 'images'), ('media', 'media'), ('score', 'score'), ('treinamento', 'treinamento'), ('sprite_strip_anim.py', '.'), ('spritesheet.py', '.')],
    hiddenimports=[],
    hookspath=[],
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
    a.binaries,
    a.datas,
    [],
    name='horse_racing_player_vs_ai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
