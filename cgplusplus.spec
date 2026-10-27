# PyInstaller build recipe for CG++.
#   pip install -r requirements-build.txt
#   pyinstaller cgplusplus.spec              -> Linux/Windows: dist/cgplusplus (.exe), a single file
#                                               macOS: dist/CG++.app
#   CG_ONEDIR=1 pyinstaller cgplusplus.spec  -> Linux/Windows: dist/cgplusplus/ folder: starts faster,
#                                               fewer antivirus flags
import os
import sys

IS_MAC = sys.platform == 'darwin'
ONEFILE = not os.environ.get('CG_ONEDIR') and not IS_MAC   # a macOS .app is always a folder inside

datas = [('opac', 'opac'), ('assets/cgplusplus.png', 'assets')]
if os.path.isfile('LICENSE'):
    datas.append(('LICENSE', '.'))

a = Analysis(
    ['cgplusplus.py'],
    datas=datas,
    excludes=[
        # only PySide6 is bundled (LGPL); keep the other Qt bindings out
        'PyQt5', 'PyQt6', 'PySide2',
        # not used by CG++
        'tkinter', '_tkinter', 'IPython', 'jupyter', 'notebook', 'pandas', 'pytest',
        'matplotlib.backends.backend_tkagg', 'matplotlib.backends._backend_tk',
        'matplotlib.backends.backend_webagg', 'matplotlib.backends.backend_nbagg',
    ],
)
pyz = PYZ(a.pure)
win_icon = 'assets/cgplusplus.ico' if sys.platform == 'win32' else None

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name='cgplusplus', icon=win_icon,
              console=False, upx=False, strip=False)
else:
    exe = EXE(pyz, a.scripts, exclude_binaries=True, name='cgplusplus', icon=win_icon,
              console=False, upx=False, strip=False)
    coll = COLLECT(exe, a.binaries, a.datas, name='cgplusplus', upx=False, strip=False)

if IS_MAC:
    app = BUNDLE(
        coll,
        name='CG++.app',
        icon='assets/cgplusplus.png',          # PyInstaller converts it to the macOS .icns format
        bundle_identifier='io.github.lisastrophys.cgplusplus',
        info_plist={
            'CFBundleDisplayName': 'CG++',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '14.0',  # set by the PySide6 and SciPy builds used
            'NSHumanReadableCopyright': 'Copyright (C) 2025 L. Zwicky. GNU GPL v3 or later.',
        },
    )
