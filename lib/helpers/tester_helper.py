import os
import tqdm
import shutil
import numpy as np

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodgp'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

        # feature saving configuration for inference
        self.save_features = cfg.get('save_features', False)
        self.feature_format = cfg.get('feature_format', 'pt')
        self.feature_save_mode = cfg.get('feature_save_mode', 'per_image')
        self.feature_output_dir = os.path.join(self.output_dir, cfg.get('feature_output_dir', 'features'))
        self.feature_buffer = {'img_ids': [], 'pred_hs': [], 'pred_depth': [], 'pred_logits': []} if self.save_features and self.feature_save_mode == 'single_file' else None

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.threshold()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            self.save_batch_features(outputs, info)

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)
        if self.save_features and self.feature_save_mode == 'single_file':
            self.save_features_file()

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result

    def save_batch_features(self, outputs, info):
        if not self.save_features:
            return

        os.makedirs(self.feature_output_dir, exist_ok=True)
        img_ids = info['img_id']
        if isinstance(img_ids, torch.Tensor):
            img_ids = img_ids.detach().cpu().tolist()

        batch_size = outputs['pred_hs'].shape[0]
        for idx in range(batch_size):
            data = {
                'pred_hs': outputs['pred_hs'][idx].detach().cpu(),
                'pred_depth': outputs['pred_depth'][idx].detach().cpu(),
                'pred_logits': outputs['pred_logits'][idx].detach().cpu(),
            }

            if self.feature_save_mode == 'single_file':
                self.feature_buffer['img_ids'].append(int(img_ids[idx]))
                self.feature_buffer['pred_hs'].append(data['pred_hs'].numpy())
                self.feature_buffer['pred_depth'].append(data['pred_depth'].numpy())
                self.feature_buffer['pred_logits'].append(data['pred_logits'].numpy())
                continue

            filename = '{:06d}.{}'.format(int(img_ids[idx]), 'pt' if self.feature_format == 'pt' else 'npz')
            output_path = os.path.join(self.feature_output_dir, filename)
            if self.feature_format == 'npz':
                np.savez_compressed(output_path,
                                    pred_hs=data['pred_hs'].numpy(),
                                    pred_depth=data['pred_depth'].numpy(),
                                    pred_logits=data['pred_logits'].numpy())
            else:
                torch.save(data, output_path)

    def save_features_file(self):
        if not self.save_features or self.feature_save_mode != 'single_file':
            return

        os.makedirs(self.feature_output_dir, exist_ok=True)
        output_path = os.path.join(self.feature_output_dir, 'features.{}'.format('pt' if self.feature_format == 'pt' else 'npz'))

        if self.feature_format == 'npz':
            np.savez_compressed(output_path,
                                img_ids=np.array(self.feature_buffer['img_ids'], dtype=np.int32),
                                pred_hs=np.stack(self.feature_buffer['pred_hs'], axis=0),
                                pred_depth=np.stack(self.feature_buffer['pred_depth'], axis=0),
                                pred_logits=np.stack(self.feature_buffer['pred_logits'], axis=0))
        else:
            torch.save(self.feature_buffer, output_path)
