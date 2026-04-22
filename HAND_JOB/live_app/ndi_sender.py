import numpy as np
from live_app.config import NDI_SOURCE_NAME

try:
    import NDIlib as ndi
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    print("[ndi_sender] ndi-python not installed — NDI output disabled.")


class NDISender:
    def __init__(self, source_name: str = NDI_SOURCE_NAME):
        self._enabled = _AVAILABLE
        self._send    = None
        if not self._enabled:
            return
        if not ndi.initialize():
            print("[ndi_sender] NDI initialize() failed — disabled.")
            self._enabled = False
            return
        settings = ndi.SendCreate()
        settings.ndi_name = source_name
        self._send = ndi.send_create(settings)
        if self._send is None:
            print("[ndi_sender] send_create() failed — disabled.")
            self._enabled = False

    def send(self, frame_bgr: np.ndarray):
        if not self._enabled or self._send is None:
            return
        h, w      = frame_bgr.shape[:2]
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        vf        = ndi.VideoFrameV2()
        vf.data          = frame_rgb
        vf.FourCC        = ndi.FOURCC_VIDEO_TYPE_RGBX
        vf.xres          = w
        vf.yres          = h
        vf.frame_rate_N  = 30
        vf.frame_rate_D  = 1
        ndi.send_send_video_v2(self._send, vf)

    def destroy(self):
        if self._enabled and self._send is not None:
            ndi.send_destroy(self._send)
            ndi.destroy()
