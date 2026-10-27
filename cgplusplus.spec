# PyInstaller build recipe for CG++.
#   pip install -r requirements-build.txt
#   pyinstaller cgplusplus.spec              -> dist/cgplusplus (.exe on Windows), a single file
#   CG_ONEDIR=1 pyinstaller cgplusplus.spec  -> dist/cgplusplus/ folder: starts faster, fewer antivirus flags
import os
import sys

ONEFILE = not os.environ.get('CG_ONEDIR')

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
icon = 'assets/cgplusplus.ico' if sys.platform == 'win32' else None

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name='cgplusplus', icon=icon,
              console=False, upx=False, strip=False)
else:
    exe = EXE(pyz, a.scripts, exclude_binaries=True, name='cgplusplus', icon=icon,
              console=False, upx=False, strip=False)
    coll = COLLECT(exe, a.binaries, a.datas, name='cgplusplus', upx=False, strip=False)
