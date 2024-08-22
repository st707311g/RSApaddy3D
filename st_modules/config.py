import logging
from typing import Final

try:
    import cupy as cp  # noqa
    from cupyx.scipy import ndimage  # noqa

    is_cupy_available: Final = True
except:  # noqa
    is_cupy_available: Final = False

AUTHOR: Final[str] = "Shota Teramoto"
COPYRIGHT: Final[str] = "2024 National Agriculture and Food Research Organization. All rights reserved."
PROGRAM_NAME: Final[str] = "RSApaddy3D"
VERSION: Final[str] = "1.0"
DESCRIPTION: Final[str] = f"{PROGRAM_NAME} (Version {VERSION}) Author: {AUTHOR}. Copyright (C) {COPYRIGHT}"

logger = logging.getLogger(PROGRAM_NAME)
