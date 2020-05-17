from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import torch.nn.functional as F

import TRNmodule

HAND_CLASS = 85

class JointTSN(nn.Module):

    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True,
                 siamese=False):
        super(JointTSN, self).__init__()

        self.model = TSN(num_class, num_segments, modality,
                     base_model=base_model, new_length=new_length,
                     consensus_type=consensus_type, before_softmax=before_softmax,
                     dropout=dropout,img_feature_dim=img_feature_dim,
                     crop_num=crop_num, partial_bn=partial_bn, print_spec=print_spec,
                     siamese=siamese)

        self.model_hand = TSN(num_class, 2, modality,
                         base_model=base_model, new_length=new_length,
                         consensus_type=consensus_type, before_softmax=before_softmax,
                         dropout=dropout, img_feature_dim=img_feature_dim,
                         crop_num=crop_num, partial_bn=partial_bn, print_spec=print_spec,
                         siamese=siamese)

        self.consensus_handshape = TRNmodule.return_TRN(consensus_type, img_feature_dim, 2, HAND_CLASS*2)
        self.agg_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*num_class, num_class)
        )
        self.pool = torch.nn.AvgPool1d(kernel_size=2)


    def forward(self, input, input_hand):

        output = self.model(input)

        # sample_len = (3 if self.model.modality == "RGB" else 2) * self.model.new_length
        #
        # if self.model.modality == 'RGBDiff':
        #     sample_len = 3 * self.model.new_length
        #     input = self.model._get_diff(input)
        #
        # # print('sample ', sample_len, 'mod '. self.modality, self.new_length)
        # base_out = self.model.base_model(input.view((-1, sample_len) + input.size()[-2:]))  # (bs*#seg, 3, 224, 224)
        # # print('base_out ', base_out.shape)
        #
        # if self.model.dropout > 0:
        #     base_out = self.model.new_fc(base_out)
        # print('base_out1 ', base_out.shape)
        #
        # if not self.model.before_softmax:
        #     base_out = self.model.softmax(base_out)
        # print('base_out2 ', base_out.shape)

        # repeat for hand
        sample_len = (3 if self.model_hand.modality == "RGB" else 2) * self.model_hand.new_length

        if self.model_hand.modality == 'RGBDiff':
            sample_len = 3 * self.model_hand.new_length
            input_hand = self.model_hand._get_diff(input_hand)

        base_out_hand = self.model_hand.base_model(input_hand.view((-1, sample_len) + input_hand.size()[-2:]))  # (bs*#seg, 3, 224, 224)

        if self.model_hand.dropout > 0:
            base_out_hand = self.model_hand.new_fc(base_out_hand)

        if not self.model_hand.before_softmax:
            base_out_hand = self.model_hand.softmax(base_out_hand) # (bs*4, 256)

        base_out_hand = base_out_hand.view((-1, 2) + base_out_hand.size()[1:])  # (bs*2, #seg, 256)
        output2 = self.model_hand.consensus(base_out_hand)  # (bs*2, #num_class)
        output_handshape = self.consensus_handshape(base_out_hand)  # (bs*2, HANDCLASS*2)

        output2 = output2.transpose(1, 0).reshape(output2.size(1), 1, -1)
        output2 = self.pool(output2).squeeze().transpose(1, 0)

        output = torch.cat((output, output2), 1)
        output = self.agg_classifier(output)

        return output.squeeze(1), output_handshape.squeeze(1)

class HandTSN(nn.Module):

    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True,
                 siamese=False):
        super(HandTSN, self).__init__()

        self.model_hand = TSN(num_class, 2, modality,
                         base_model=base_model, new_length=new_length,
                         consensus_type=consensus_type, before_softmax=before_softmax,
                         dropout=dropout, img_feature_dim=img_feature_dim,
                         crop_num=crop_num, partial_bn=partial_bn, print_spec=print_spec,
                         siamese=siamese)

        self.consensus_handshape = TRNmodule.return_TRN(consensus_type, img_feature_dim, 2, HAND_CLASS*2)
        self.pool = torch.nn.AvgPool1d(kernel_size=2)


    def forward(self, input_hand):

        # sample_len = (3 if self.model.modality == "RGB" else 2) * self.model.new_length
        #
        # if self.model.modality == 'RGBDiff':
        #     sample_len = 3 * self.model.new_length
        #     input = self.model._get_diff(input)
        #
        # # print('sample ', sample_len, 'mod '. self.modality, self.new_length)
        # base_out = self.model.base_model(input.view((-1, sample_len) + input.size()[-2:]))  # (bs*#seg, 3, 224, 224)
        # # print('base_out ', base_out.shape)
        #
        # if self.model.dropout > 0:
        #     base_out = self.model.new_fc(base_out)
        # print('base_out1 ', base_out.shape)
        #
        # if not self.model.before_softmax:
        #     base_out = self.model.softmax(base_out)
        # print('base_out2 ', base_out.shape)

        # repeat for hand
        sample_len = (3 if self.model_hand.modality == "RGB" else 2) * self.model_hand.new_length

        if self.model_hand.modality == 'RGBDiff':
            sample_len = 3 * self.model_hand.new_length
            input_hand = self.model_hand._get_diff(input_hand)

        base_out_hand = self.model_hand.base_model(input_hand.view((-1, sample_len) + input_hand.size()[-2:]))  # (bs*#seg, 3, 224, 224)

        if self.model_hand.dropout > 0:
            base_out_hand = self.model_hand.new_fc(base_out_hand)

        if not self.model_hand.before_softmax:
            base_out_hand = self.model_hand.softmax(base_out_hand) # (bs*4, 256)

        base_out_hand = base_out_hand.view((-1, 2) + base_out_hand.size()[1:])  # (bs*2, #seg, 256)
        output2 = self.model_hand.consensus(base_out_hand)  # (bs*2, #num_class)
        output_handshape = self.consensus_handshape(base_out_hand)  # (bs*2, HANDCLASS*2)

        output2 = output2.transpose(1, 0).reshape(output2.size(1), 1, -1)
        output2 = self.pool(output2).squeeze().transpose(1, 0)


        return output2.squeeze(1), output_handshape.squeeze(1)




class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True,
                 siamese=False, hand=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.siamese = siamese
        self.hand = hand
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 3
        else:
            self.new_length = new_length

        if self.modality == 'RGB':
            self.new_length = 1

        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow': # TODO hand
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        if consensus_type in ['TRN', 'TRNmultiscale']:
            # plug in the Temporal Relation Network Module
            self.consensus = TRNmodule.return_TRN(consensus_type, self.img_feature_dim, self.num_segments, num_class)
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if self.siamese:
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104,117,128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1+self.new_length)

        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward_once(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        #print('sample ', sample_len, 'mod '. self.modality, self.new_length)
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:])) # (bs*#seg, 3, 224, 224)
        #print('base_out ', base_out.shape)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        #print('base_out1 ', base_out.shape)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        #print('base_out2 ', base_out.shape)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])  # (bs, #seg, 256)
        #exit()

        output = self.consensus(base_out)  # (bs, #num_class)
        return output.squeeze(1)

    def forward(self, input, input2=None):
        output = self.forward_once(input)
        if input2 is not None:
            output2 = self.forward_once(input2)
            output = self.cos(output, output2)

        return output


    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)

        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
