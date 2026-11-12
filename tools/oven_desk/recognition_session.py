"""UI request ownership and result storage; never model preprocessing or inference."""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from capture_core import now_local, safe_stem, write_exclusive
from inference_engine import file_sha256

TIMING_SCOPE='预处理、推理、输出检查（含可能的模型锁等待；不含图片解码、模型加载、文件哈希与界面绘制）'


def frame_sha256(frame):
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or not frame.size:
        raise ValueError('需要有效的三通道图像')
    checksum=hashlib.sha256(f'{frame.dtype}:{frame.shape}'.encode('ascii'))
    checksum.update(np.ascontiguousarray(frame).tobytes())
    return checksum.hexdigest()


@dataclass(frozen=True,eq=False)
class PredictionRequest:
    request_id: int
    source_generation: int
    model_generation: int
    frame: np.ndarray
    frame_sha256: str
    source: dict
    model_identity: dict
    started_at: str


class RecognitionSession:
    def __init__(self):
        self.source_generation=0
        self.model_generation=0
        self.sequence=0
        self.source={'kind':'camera','label':'摄像头未连接','original_sha256':None}
        self.source_frame=None
        self.active=None
        self.record=None
        self.error=''

    def set_source(self,kind,label,frame=None,original_sha256=None):
        if kind not in {'file','camera'}:raise ValueError('无效的图像来源')
        if frame is not None:frame_sha256(frame)
        self.source_generation+=1
        self.source={'kind':kind,'label':str(label),'original_sha256':original_sha256}
        self.source_frame=frame.copy() if frame is not None else None
        self.record=None;self.error=''

    def invalidate_model(self):
        self.model_generation+=1
        self.record=None;self.error=''

    def begin(self,frame,model_identity):
        if self.active is not None:raise RuntimeError('识别进行中，请稍后')
        sha=frame_sha256(frame)
        if not isinstance(model_identity,dict) or not model_identity.get('model_sha256'):
            raise ValueError('模型身份尚未验证')
        owned=np.ascontiguousarray(frame).copy();owned.flags.writeable=False
        self.sequence+=1
        request=PredictionRequest(self.sequence,self.source_generation,self.model_generation,owned,sha,
                                  copy.deepcopy(self.source),copy.deepcopy(model_identity),now_local())
        self.active=request;self.record=None;self.error=''
        return request

    def _owns(self,request):
        return self.active is request

    def _current(self,request):
        return request.source_generation==self.source_generation and request.model_generation==self.model_generation

    def finish(self,request,result,latency_ms):
        if not self._owns(request):return False
        if not self._current(request):
            self.active=None
            return False
        if type(latency_ms) not in (int,float) or not math.isfinite(latency_ms) or latency_ms<0:
            raise ValueError('识别耗时必须是实际测量的有限非负值')
        if frame_sha256(request.frame)!=request.frame_sha256:raise ValueError('识别帧已改变')
        self.record=dict(request_id=request.request_id,source=copy.deepcopy(request.source),
            model=copy.deepcopy(request.model_identity),frame_sha256=request.frame_sha256,
            started_at=request.started_at,finished_at=now_local(),latency_ms=float(latency_ms),
            timing_scope=TIMING_SCOPE,prediction=copy.deepcopy(result),frame=request.frame)
        self.active=None;self.error=''
        return True

    def fail(self,request,reason):
        if not self._owns(request):return False
        self.active=None
        if not self._current(request):return False
        self.record=None;self.error=str(reason)
        return True

    def save_result(self,folder):
        if self.record is None or self.active is not None:raise ValueError('没有可保存的当前识别结果')
        if not str(folder).strip():raise ValueError('请选择结果保存目录')
        record=self.record;frame=record['frame']
        if frame_sha256(frame)!=record['frame_sha256']:raise ValueError('识别帧完整性检查失败')
        ok,encoded=cv2.imencode('.png',frame)
        if not ok:raise ValueError('无法编码识别帧')
        parent=Path(folder).expanduser();parent.mkdir(parents=True,exist_ok=True)
        target=parent/safe_stem('recognition');target.mkdir(exist_ok=False)
        image=target/'frame.png';metadata=target/'result.json'
        try:
            write_exclusive(image,[encoded.tobytes()])
            saved={key:value for key,value in record.items() if key!='frame'}
            saved.update(format='het-oven-recognition-result/v1',saved_png_sha256=file_sha256(image),
                         frame_size=[frame.shape[1],frame.shape[0]],frame_format='decoded BGR uint8, saved lossless PNG',
                         scope='model scores only; no ground truth or independent accuracy')
            write_exclusive(metadata,[(json.dumps(saved,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')])
        except Exception:
            # Only these files in this exclusively created bundle belong to this operation.
            image.unlink(missing_ok=True);metadata.unlink(missing_ok=True);target.rmdir()
            raise
        return target
