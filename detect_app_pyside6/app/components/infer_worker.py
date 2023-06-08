# coding:utf-8
import base64
import copy
import os
from copy import deepcopy
from queue import Queue
from typing import Tuple, Sequence, Dict, Union, List, Any

import cv2
import mmengine
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from PySide6.QtCore import QThread, QObject, Signal, Slot, qDebug
from PySide6.QtGui import QImage, QPixmap

from mmdeploy_runtime import Detector
from mmcv.ops import batched_nms

from sahi.slicing import slice_image, shift_bboxes

from ..utils import ndarray_to_qpixmap



class InferWorker(QObject):
    task_complete_signal = Signal(QPixmap)
    task_log_signal = Signal(str)
    task_under_dealing_signal = Signal(QPixmap)
    task_finished_signal = Signal(str)

    batch_size = 4
    patch_size = 1000

    def __init__(self, consumer_image_queue: Queue, productor_det_record_queue: Queue):
        super().__init__()
        self.image_queue = consumer_image_queue
        self.det_record_queue = productor_det_record_queue

        self.is_continue = True

        self.detector = Detector(
            model_path='resource/model-onnx', device_name='cpu', device_id=0)


    @Slot()
    def quit_slot(self):
        self.is_continue = False
        self.task_log_signal.emit('Set Quit Flag on Infer Worker')


    def shift_predictions(self, det_data_samples: list,
                          offsets: Sequence[Tuple[int, int]],
                          src_image_shape: Tuple[int, int]):

        shifted_predictions = []
        for det_sample, offset in zip(det_data_samples, offsets):
            pred_inst = deepcopy(det_sample)

            # Check bbox type
            if pred_inst['bboxes'][-1].size == 4:
                # Horizontal bboxes
                shifted_bboxes = shift_bboxes(pred_inst['bboxes'], offset)
            else:
                raise NotImplementedError

            # shift bboxes
            pred_inst['bboxes'] = shifted_bboxes

            shifted_predictions.append(deepcopy(pred_inst))

        return shifted_predictions

        # shifted_predictions = InstanceData.cat(shifted_predictions)

    def merge_bbox_by_kd(self, bboxes: List, labels: List):
        labeled_bboxes = dict()
        for idx, label in enumerate(labels):
            if bboxes[idx][4] > 0.1:
                if label in labeled_bboxes.keys():
                    labeled_bboxes[label].append(bboxes[idx])
                else:
                    labeled_bboxes[label] = [bboxes[idx]]
        merged_bboxes = []
        merged_labels = []
        if len(labeled_bboxes) == 0:
            return
        for label_cls in labeled_bboxes.keys():
            sorted_boxes = deepcopy(labeled_bboxes[label_cls])

            rects = []
            rectsUsed = []
            for box in sorted_boxes:
                rects.append(box)
                rectsUsed.append(False)

            rects.sort(key=lambda x: x[0])
            acceptedRects = []
            xThr = 0.5

            for idx, val in enumerate(rects):
                if not rectsUsed[idx]:

                    currxMin = val[0]
                    currxMax = val[2]
                    curryMin = val[1]
                    curryMax = val[3]

                    rectsUsed[idx] = True
                    for subIdx, subVal in enumerate(rects[(idx + 1):], start=(idx + 1)):
                        candixMin = subVal[0]
                        candixMax = subVal[2]
                        candiyMin = subVal[1]
                        candiyMax = subVal[3]

                        if candixMin <= currxMax + xThr:
                            currxMax = candixMax
                            curryMin = min(curryMin, candiyMin)
                            curryMax = max(curryMax, candiyMax)

                            rectsUsed[subIdx] = True
                        else:
                            break
                    acceptedRects.append(
                        [currxMin, curryMin, currxMax, curryMax, val[-1] if val[-1] > subVal[-1] else subVal[-1]])

            merged_labels.extend(np.ones(len(acceptedRects)).tolist())
            merged_bboxes.extend(acceptedRects)

        return dict(bboxes=[bbox[0:4] for bbox in merged_bboxes], scores=[bbox[-1] for bbox in merged_bboxes],
                    labels=merged_labels)

    def merge_results_by_nms(self,
                             results: list,
                             offsets: Sequence[Tuple[int, int]],
                             src_image_shape: Tuple[int, int],
                             nms_cfg: dict) -> Dict[str, Union[List[List[Union[np.ndarray, Any]]], List[Any]]]:
        shifted_instances = self.shift_predictions(results, offsets, src_image_shape)

        scores = torch.Tensor(np.concatenate([ins['scores'] for ins in shifted_instances]))
        labels = torch.Tensor(np.concatenate([ins['labels'] for ins in shifted_instances]))
        bboxes = torch.Tensor(np.concatenate([ins['bboxes'] for ins in shifted_instances]))

        bboxes_ret, keeps = batched_nms(boxes=bboxes, scores=scores, idxs=labels, nms_cfg=nms_cfg)

        # merged_result = dict(bboxes=bboxes[keeps], labels=labels[keeps], scores=scores[keeps])
        merged_result = self.merge_bbox_by_kd(bboxes_ret.numpy().tolist(),
                                              labels[keeps].numpy().astype(np.int32).tolist())
        return merged_result

    def do_inference(self) -> None:
        self.task_log_signal.emit('Starting inference......')

        while self.is_continue:
            image = self.image_queue.get(block=True)
            image = np.array(image)
            raw_img = copy.deepcopy(image)
            height, width, channels = image.shape
            print('{}-{}-{}'.format(height, width, channels))

            self.task_under_dealing_signal.emit(ndarray_to_qpixmap(deepcopy(image)))
            self.task_log_signal.emit('Get image from queue......')

            assert image is not None
            sliced_image_object = slice_image(
                image,
                slice_height=self.patch_size,
                slice_width=self.patch_size,
                auto_slice_resolution=False,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
            )

            start = 0
            bbox_result = []
            while True:
                end = min(start + self.batch_size, len(sliced_image_object))
                images = []
                for sliced_image in sliced_image_object.images[start:end]:
                    images.append(sliced_image)

                # bboxes, labels, masks = detector(np.array(images))
                det_res = self.detector.batch(images)
                bbox_result.extend(
                    [dict(bboxes=det_record[0][:, :4], labels=det_record[1], scores=det_record[0][:, -1]) for
                     det_record in
                     det_res])

                if end >= len(sliced_image_object):
                    break
                start += self.batch_size

            assert bbox_result is not None
            image_result = self.merge_results_by_nms(
                results=bbox_result,
                offsets=sliced_image_object.starting_pixels,
                src_image_shape=(height, width),
                nms_cfg={
                    'type': 'nms',
                    'iou_threshold': 0.2
                })

            palette = np.random.randint(0, 256, size=(len(sliced_image_object.starting_pixels), 3))
            palette = [tuple(c.astype(np.int32).tolist()) for c in palette]

            for window_id, starting_pixel in enumerate(sliced_image_object.starting_pixels):
                cv2.rectangle(image,
                              starting_pixel,
                              (np.array(starting_pixel) + self.patch_size).astype(np.int32).tolist(),
                              color=palette[window_id], thickness=1)

            print('labels:{}'.format( image_result['labels']))
            print('scores:{}'.format( image_result['scores']))

            res = dict(labels=[], bbox=[])
            for bbox, label, score in zip(
                    image_result['bboxes'],
                    image_result['labels'],
                    image_result['scores']):
                # if score > 0.1:
                left = tuple([int(bbox[0]), int(bbox[1])])
                bottom = tuple([int(bbox[2]), int(bbox[3])])
                assert image is not None
                print('L:{} R:{} W: {},  H: {}, score: {}'.format(left, bottom, bbox[2] - bbox[0],
                                                                  bbox[3] - bbox[1],
                                                                  score))
                res['labels'].append(label)
                res['bbox'].append(bbox)

                cv2.putText(image, str(int(label)), [ix if ix - 10 < 0 else ix for ix in left], cv2.FONT_HERSHEY_PLAIN, 2, color=(0, 255, 255), thickness=4)
                cv2.rectangle(image, left, bottom, color=(0, 0, 255), thickness=2)

            self.task_complete_signal.emit(ndarray_to_qpixmap(image))
            success, encoded_image = cv2.imencode('.jpg', raw_img)


            # 将字节流转换为 bytes
            # base64_str = base64.b64encode(encoded_image).decode('utf-8')
            if len(image_result['bboxes']) > 0:
                self.det_record_queue.put(
                    dict(img=encoded_image,
                         izPass=len(res['labels']) == 0,
                         categories=[int(cate_id) for cate_id in image_result['labels']],
                         image_height=image.shape[0],
                         image_width=image.shape[1],
                         bbox=res['bbox']
                    )
                )

            QThread.sleep(1)

        self.task_log_signal.emit('Task Complete!')
        self.task_finished_signal.emit('Okk')


