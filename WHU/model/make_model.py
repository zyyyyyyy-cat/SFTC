import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup
from .backbones.sftc_local import SFTCSource, sensor_order, validate_sftc_config

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.sftc_enabled = cfg.SFTC.ENABLED
        if self.sftc_enabled:
            validate_sftc_config(cfg)
            self.sensor_order = sensor_order(cfg.DATASETS.MODALITIES)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path, strict_backbone=self.sftc_enabled)
            print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if self.sftc_enabled:
            self.base.sftc_block_index = cfg.SFTC.BLOCK_INDEX
            attention = self.base.blocks[self.base.sftc_block_index].attn
            self.base.sftc = SFTCSource(
                attention.qkv, attention.num_heads, cfg.SFTC.POS_SCALE,
                cfg.SFTC.SOURCE_BIAS_INIT, cfg.SFTC.SOURCE_BIAS_FIXED_EPOCHS)
            print('SFTC sensor mapping (folder -> RGB/NIR/TIR row): {}'.format(
                dict(zip(cfg.DATASETS.MODALITIES, self.sensor_order))))

    def training_modal_ids(self, images):
        if len(images) != len(self.sensor_order):
            raise ValueError('Input modality chunks must follow DATASETS.MODALITIES')
        return torch.cat([torch.full((image.shape[0],), sensor, dtype=torch.long,
                                     device=image.device)
                          for image, sensor in zip(images, self.sensor_order)])

    def evaluation_modal_ids(self, dataset_ids, device):
        # WHU_MARS enumerates configured folders starting at one.
        ids = torch.as_tensor(dataset_ids, dtype=torch.long, device=device)
        if ids.ndim != 1 or bool(((ids < 1) | (ids > len(self.sensor_order))).any()):
            raise ValueError('WHU dataset modality IDs must be one-based folder indices')
        mapping = ids.new_tensor(self.sensor_order)
        return mapping[ids - 1]
        
    def forward(self, x=None, label=None, camids=None, mode=0, modal_ids=None, sftc_enabled=True):
        if mode==0:
            if self.sftc_enabled and modal_ids is None:
                modal_ids = self.training_modal_ids(x)
            x = torch.cat(list(x), dim=0)
            global_feat = self.base(x, modal_label=modal_ids, sftc_enabled=sftc_enabled)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, feat
        else:
            global_feat = self.base(x, modal_label=modal_ids, sftc_enabled=sftc_enabled)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        if self.sftc_enabled:
            self._load_sftc_checkpoint(trained_path, skip_classifier=True)
            return
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        if self.sftc_enabled:
            self._load_sftc_checkpoint(model_path, skip_classifier=False)
            return
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def _load_sftc_checkpoint(self, path, skip_classifier):
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state:
            state = state['model']
        state = {(key[7:] if key.startswith('module.') else key): value
                 for key, value in state.items()}
        if skip_classifier:
            state = {key: value for key, value in state.items() if not key.startswith('classifier.')}
        expected = set(self.state_dict())
        if skip_classifier:
            expected = {key for key in expected if not key.startswith('classifier.')}
        missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError('SFTC checkpoint mismatch: missing={}, unexpected={}. '
                               'Use a full SFTC checkpoint for evaluation.'.format(missing, unexpected))
        self.load_state_dict(state, strict=not skip_classifier)
        print('Loaded full SFTC checkpoint from {}; local K/V preserved'.format(path))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
