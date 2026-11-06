import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inference_engine import (ModelContractError, load_labels, prepare_input,
    parse_classification, parse_yolo_detection, read_image_unicode)

class InferTests(unittest.TestCase):
    def test_label_lists(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / '食材.json'
            p.write_text(json.dumps({'classes':[{'id':0,'key':'corn','zh':'玉米'},
                                              {'id':1,'key':'carrot','zh':'胡萝卜'}]}, ensure_ascii=False), encoding='utf-8')
            self.assertEqual(load_labels(p), ['玉米','胡萝卜'])
            p.write_text('{"classes":[{"id":1,"key":"corn"}]}', encoding='utf-8')
            with self.assertRaises(ModelContractError): load_labels(p)

    def test_classification_probabilities_and_logits(self):
        p = parse_classification(np.array([[.1,.8,.1]], dtype=np.float32), ['a','b','c'])
        self.assertEqual(p[0]['label'],'b')
        self.assertAlmostEqual(p[0]['score'],.8,places=5)
        q = parse_classification(np.array([[2,6,3]], dtype=np.float32), ['a','b','c'])
        self.assertEqual(q[0]['id'],1)
        self.assertAlmostEqual(sum(x['score'] for x in q),1)
        with self.assertRaises(ModelContractError):
            parse_classification(np.array([[.1,np.nan]]), ['a','b'])

    def test_yolo_class_aware_boxes(self):
        preds = np.zeros((1,6,2), dtype=np.float32)
        preds[0,:,0] = [32,32,20,20,.9,.05]
        preds[0,:,1] = [32,32,20,20,.05,.8]
        context={"gain":1.,"pad_left":0,"pad_top":0,"orig_w":64,"orig_h":64}
        out = parse_yolo_detection(preds,['potato','carrot'],context)
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]['label'],'potato')
        self.assertEqual(out[1]['label'],'carrot')
        with self.assertRaises(ModelContractError):
            parse_yolo_detection(np.zeros((1,84,20)), ['potato','carrot'],context)

    def test_preprocessing_and_chinese_images(self):
        frame=np.zeros((108,192,3), dtype=np.uint8)
        frame[:,:,:]=[50,100,200]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'照片-玉米.png'
            ok, enc=cv2.imencode('.png',frame)
            self.assertTrue(ok); p.write_bytes(enc.tobytes())
            np.testing.assert_array_equal(read_image_unicode(p),frame)
        x,context=prepare_input(frame,'yolo_detect',320,320)
        self.assertEqual(x.shape,(1,3,320,320))
        self.assertEqual(context['orig_w'],192)
        y,_=prepare_input(frame,'efficientnet_imagenet',224,224)
        self.assertEqual(y.shape,(1,3,224,224))

if __name__=='__main__': unittest.main()
