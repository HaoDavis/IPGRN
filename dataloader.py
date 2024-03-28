import os
import PIL
import torch
from PIL.Image import Image
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from radiomics import featureextractor
import torchvision.models.video as models
import torchvision.transforms as transforms
from torchvision import models as tv_models
from torch.autograd import Variable
import torch.nn as nn


def pickle_data(data_set, data_path: str, folds: list, name_list: list, out_name="cbbct_feat"):
    # 写每个患者的pkl文件, 防卡死
    data_set_len = len(data_set)
    pkl_count = 0
    if not os.path.exists(f"{data_path}/pickles"):
        os.makedirs(f"{data_path}/pickles")
    for fold in folds:  # ['2012', '2013', '2019', '2020']
        fold_path = f"{data_path}/{fold}"
        if os.path.exists(fold_path):
            for image_name in os.listdir(fold_path):
                if image_name in name_list:
                    pkl_count += 1
                    index = name_list.index(image_name)
                    print(f"\r{image_name}\t{pkl_count}/{data_set_len}", end='', flush=True)
                    with open(f"{data_path}/pickles/{image_name}.pkl", 'wb') as f:
                        pickle.dump(data_set[index], f)
    print("\n有效处理", pkl_count, "个样本\n汇聚中...")
    # 读每个患者的pkl文件
    data_list = []
    for i, pkl_name in enumerate(os.listdir(f"{data_path}/pickles")):
        print(f"\r{pkl_name}\t{i + 1}/{pkl_count}", end='', flush=True)
        with open(f'{data_path}/pickles/{pkl_name}', 'rb') as f:
            data_list.append(pickle.load(f))
    # stack之后再汇成一个pkl文件
    stacked = []
    for j in range(len(data_list[0])):
        stacked.append(torch.stack([data_list[i][j] for i in range(len(data_list))]))
    with open(f'{data_path}/{out_name}.pkl', 'wb') as f:
        pickle.dump(stacked, f)
    print("\rdone.")


class CBBCTDataset(Dataset):

    def __init__(self, root_dir):  # train_split=0.8, val_split=0.1, test_split=0.1, random_seed=42
        self.root_dir = root_dir
        excel_path = os.path.join(root_dir, 'Clinical_Information.xlsx')
        self.excel_file = pd.read_excel(excel_path, header=0)
        self.image_2012 = os.path.join(root_dir, '2012')
        self.image_2013 = os.path.join(root_dir, '2013')
        self.image_2019 = os.path.join(root_dir, '2019')
        self.image_2020 = os.path.join(root_dir, '2020')
        self.all_labels = [[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2, 3, 4, 5],
                           [0, 1, 2, 3],
                           [0, 1, 2],
                           [0, 1, 2],
                           [0, 1],
                           [0, 1],
                           [0, 1, 2, 3],
                           [0, 1]]
        # train_data, temp_data = train_test_split(self.excel_file, train_size=train_split, random_state=random_seed)
        # val_data, test_data = train_test_split(temp_data, train_size=val_split / (val_split + test_split),
        #                                        random_state=random_seed)
        # self.train_data = train_data
        # self.val_data = val_data
        # self.test_data = test_data

    def __getitem__(self, index):
        image_id = str(int(self.excel_file.values[index][0]))
        clinical_information = self.excel_file.values[index, 1:3]
        target_label = self.excel_file.values[index][3:13]
        clinical_information_tensor = torch.tensor(clinical_information)
        target_label_tensor = torch.tensor(target_label)
        # radiomics
        if os.path.exists(os.path.join(self.image_2012, image_id)):
            image_path = os.path.join(self.image_2012, image_id + '/data.nii.gz')
            mask_path = os.path.join(self.image_2012, image_id + '/mask.nii.gz')
        elif os.path.exists(os.path.join(self.image_2013, image_id)):
            image_path = os.path.join(self.image_2013, image_id + '/data.nii.gz')
            mask_path = os.path.join(self.image_2013, image_id + '/mask.nii.gz')
        elif os.path.exists(os.path.join(self.image_2019, image_id)):
            image_path = os.path.join(self.image_2019, image_id + '/data.nii.gz')
            mask_path = os.path.join(self.image_2019, image_id + '/mask.nii.gz')
        elif os.path.exists(os.path.join(self.image_2020, image_id)):
            image_path = os.path.join(self.image_2020, image_id + '/data.nii.gz')
            mask_path = os.path.join(self.image_2020, image_id + '/mask.nii.gz')
        else:
            raise FileNotFoundError(f"编号 {image_id} 对应的图片文件未找到。")

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        if image.GetSize() != mask.GetSize():
            # print("Input sizes are not matching, trying to transpose...")
            mask_np = sitk.GetArrayFromImage(mask)
            transposed_mask_np = np.transpose(mask_np)
            mask = sitk.GetImageFromArray(transposed_mask_np)

        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute(image, mask)

        # 获取所有以"original_"开头的特征
        original_features = {}
        for key in result.keys():
            if key.startswith("original_"):
                original_features[key] = result[key]

        feature_list = list(original_features.values())
        feature_vector = np.array(feature_list)
        radiomics_feature = torch.tensor(feature_vector)

        # image
        image_3d = nib.load(image_path).get_fdata()
        model = models.r3d_18(pretrained=True)
        model.eval()
        model = model.cuda()
        # 调整大小为(64, 64, 64)
        resize = transforms.Resize((64, 64))
        crop = transforms.CenterCrop((64, 64))
        to_tensor = transforms.ToTensor()
        # 将图像数据转换为PyTorch张量，并调整维度顺序为(C, D, H, W)
        print(image_3d.shape)
        image_3d = torch.from_numpy(image_3d).cuda()
        print(image_3d.shape)
        image_3d = resize(image_3d)
        print(image_3d.shape)
        image_3d = crop(image_3d)
        print(image_3d.shape)
        image_data = image_3d.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        # 复制通道以适应3D ResNet的输入
        image_data = image_data.repeat(1, 3, 1, 1, 1).float()
        print(image_data.shape)
        # 利用3D ResNet提取特征
        with torch.no_grad():
            image_features = model(image_data)
        # # 对齐
        # feature_dim = image_features.shape[-1]
        # padding_size1 = feature_dim - clinical_information_tensor.shape[-1]
        # padding_size2 = feature_dim - radiomics_feature.shape[-1]
        # clinical_information_tensor = F.pad(clinical_information_tensor, (0, padding_size1))
        # radiomics_feature = F.pad(radiomics_feature, (0, padding_size2))
        return clinical_information_tensor, radiomics_feature, image_features, target_label_tensor

    def __len__(self):
        return len(self.excel_file)


# 修改1 读取pickle的特征 代替CBBCTDataset 载入会快些
class CBBCTDataset2(Dataset):

    def __init__(self, root_dir, out_dims=None, aliquot=3*5, cuda=True):
        self.name = 'cbbct'
        # 读取pkl文件，整理tensor，stacked[-1]是label
        with open(f'{root_dir}/cbbct_feat.pkl', 'rb') as f:
            stacked = pickle.load(f)
        stacked[0] = stacked[0].float()
        stacked[1] = stacked[1].float()
        stacked[2] = stacked[2][:, 0, :]  # 原本是(batch_size, 1, 400)，变为(batch_size, 400)
        stacked = [me.cuda() if cuda else me.cpu() for me in stacked]  # 对每个成员me选择运行设备

        # 降维
        if out_dims is not None:
            out_dims = out_dims if isinstance(out_dims, list) else [out_dims]*len(stacked[:-1])
            print("features' shape before lstm:", [me.shape for me in stacked[:-1]])  # 最后一个是label，不用考虑
            for i, me in enumerate(stacked[:-1]):
                if out_dims[i] <= 0:
                    continue
                me_lstm = nn.LSTM(input_size=me.shape[1],
                                  hidden_size=int(me.shape[1] * out_dims[i] / 2) if isinstance(out_dims[i], float) else out_dims[i] // 2,
                                  num_layers=2, bidirectional=True, dropout=0.5)
                if cuda:
                    me_lstm = me_lstm.cuda()
                me_lstm.requires_grad_(False)
                stacked[i] = me_lstm(me)[0]
            print("features' shape after lstm:", [me.shape for me in stacked[:-1]])

        # 样本返回格式
        self.data = []
        for i in range(stacked[0].shape[0]):  # i：样本数
            self.data.append(tuple(stacked[j][i] for j in range(len(stacked))))  # j：特征数
        self.data = self.data[:len(self.data) // aliquot * aliquot]  # 去除部分样本 使其能被batch_size整除且按比例划分数据集
        self.all_labels = [[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2, 3, 4, 5],
                           [0, 1, 2, 3],
                           [0, 1, 2],
                           [0, 1, 2],
                           [0, 1],
                           [0, 1],
                           [0, 1, 2, 3],
                           [0, 1]]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class IEMOCAPDataset(Dataset):  # 读取IEMOCAP_features的Dataset

    def __init__(self, root_dir, pkl_name="IEMOCAP_features", out_dims=64, task_lim=10, aliquot=3*5, cuda=True):
        self.name = pkl_name
        # 读取pkl文件，整理tensor
        # 让模态数据从(batch_szie, seq_len, m)转为(batch_szie*seq_len, m)
        # 让label从(batch_szie, seq_len)转为(batch_szie*seq_len, 1)
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence = \
            pickle.load(open(f'{root_dir}/{pkl_name}.pkl', 'rb'), encoding='latin1')[:7]
        data = []
        for vid in videoIDs.keys():
            data.append((torch.FloatTensor(np.array(videoText[vid])),
                         torch.FloatTensor(np.array(videoVisual[vid])),
                         torch.FloatTensor(np.array(videoAudio[vid])),
                         torch.LongTensor(np.array(videoLabels[vid]))))
        data = pd.DataFrame(data)
        data_cated = [torch.cat([cell_data for cell_data in data[i]]) for i in data]
        train_textf, train_visuf, train_acouf, train_label = [d.cuda() for d in data_cated] if cuda else data_cated
        train_label = train_label.unsqueeze(1)

        # task_lim种分类
        labels = []
        for value in videoLabels.values():
            labels += value
        self.all_labels = [list(set(labels)) for _ in range(task_lim)]

        # 将相邻的task_lim个sequence缝在一起作为一个大sequence 每个sequence都有task_lim个分类
        members = [train_textf, train_visuf, train_acouf, train_label]
        for i, me in enumerate(members):
            me = me[:len(train_label) // task_lim * task_lim]  # 丢弃剩余除不尽的部分
            members[i] = me.reshape(me.shape[0]//task_lim, me.shape[1]*task_lim)
        train_textf, train_visuf, train_acouf, train_label = members

        # 降维
        print("shape before lstm:", train_textf.shape, train_visuf.shape, train_acouf.shape)
        members = [train_textf, train_visuf, train_acouf]
        for i, me in enumerate(members):
            me_lstm = nn.LSTM(input_size=me.shape[1],
                              hidden_size=int(me.shape[1] * out_dims / 2) if isinstance(out_dims, float) else out_dims // 2,
                              num_layers=2, bidirectional=True, dropout=0.5)
            if cuda:
                me_lstm = me_lstm.cuda()
            me_lstm.requires_grad_(False)
            members[i] = me_lstm(me)[0]
        train_textf, train_visuf, train_acouf = members
        print("shape after lstm:", train_textf.shape, train_visuf.shape, train_acouf.shape)  # (seq, batch, m)

        # 样本返回格式
        self.data = [(train_textf[i], train_visuf[i], train_acouf[i], train_label[i]) for i in range(len(train_label))]
        self.data = self.data[:len(self.data) // aliquot * aliquot]  # 去除部分样本 使其能被batch_size整除且按比例划分数据集

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ADNCDataset(Dataset):
    class Pkl(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            excel_path = os.path.join(root_dir, 'ADNC.csv')
            self.excel_file = pd.read_csv(excel_path, header=0)
            self.img_dir_list = ['AD', 'NC']

        def __getitem__(self, index):
            image_id = self.excel_file.values[index][0]
            target_label = self.excel_file.values[index][2]
            target_label_tensor = torch.tensor(target_label)
            image_path = None
            for img_dir in self.img_dir_list:
                if os.path.exists(os.path.join(self.root_dir, img_dir, image_id)):
                    image_path = os.path.join(self.root_dir, img_dir, image_id)
            if image_path is None:
                raise FileNotFoundError(f"编号 {image_id} 对应的图片文件未找到。")

            image_3d = nib.load(image_path).get_fdata()
            model = models.r3d_18(pretrained=True)
            model.eval()
            model = model.cuda()
            # 调整大小为(64, 64, 64)
            resize = transforms.Resize((64, 64))
            crop = transforms.CenterCrop((64, 64))
            to_tensor = transforms.ToTensor()
            # 将图像数据转换为PyTorch张量，并调整维度顺序为(C, D, H, W)
            image_3d = torch.from_numpy(image_3d).cuda()
            image_3d = resize(image_3d)
            image_3d = crop(image_3d)
            image_data = image_3d.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            # 复制通道以适应3D ResNet的输入
            image_data = image_data.repeat(1, 3, 1, 1, 1).float()
            # 利用3D ResNet提取特征
            with torch.no_grad():
                image_features = model(image_data)
            return image_features, target_label_tensor

        def __len__(self):
            return len(self.excel_file)

    def __init__(self, root_dir, pkl_name="adnc_features", out_dims=64, task_lim=1, aliquot=3*5, cuda=True):
        self.name = pkl_name
        # 读取pkl文件，整理tensor，stacked[-1]是标签值
        with open(f'{root_dir}/{pkl_name}.pkl', 'rb') as f:
            stacked = pickle.load(f)
        stacked[0] = stacked[0][:, 0, :]
        stacked[1] = stacked[1].unsqueeze(1)
        stacked = [me.cuda() if cuda else me.cpu() for me in stacked]

        # 乘以task_lim种分类
        output_tran = stacked[-1].transpose(0, 1)
        labels = [set(output_line.tolist()) for output_line in output_tran]
        self.all_labels = [list(lab) for _ in range(task_lim) for lab in labels]

        # 将相邻的task_lim个样本缝在一起 每个样本都有task_lim个分类
        sample_count = stacked[0].shape[0] // task_lim  # 丢弃剩余除不尽的部分
        for i, me in enumerate(stacked):
            me = me[:sample_count * task_lim]
            stacked[i] = me.reshape(me.shape[0]//task_lim, me.shape[1]*task_lim)

        # 降维
        if out_dims > 0:
            print("features' shape before lstm:", [me.shape for me in stacked[:-1]])
            for i, me in enumerate(stacked[:-1]):
                me_lstm = nn.LSTM(input_size=me.shape[1],
                                  hidden_size=int(me.shape[1] * out_dims / 2) if isinstance(out_dims, float) else out_dims // 2,
                                  num_layers=2, bidirectional=True, dropout=0.5)
                if cuda:
                    me_lstm = me_lstm.cuda()
                me_lstm.requires_grad_(False)
                stacked[i] = me_lstm(me)[0]
            print("features' shape after lstm:", [me.shape for me in stacked[:-1]])

        # 样本返回格式
        self.data = []
        for i in range(sample_count):  # i：样本数
            self.data.append(tuple(stacked[j][i].cuda() for j in range(len(stacked))))  # j：特征数
        self.data = self.data[:len(self.data) // aliquot * aliquot]  # 去除部分样本 使其能被batch_size整除且按比例划分数据集

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BloodCellDataset(Dataset):
    class Pkl(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.relative_dir_list = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
            self.img_name_list = []
            self.relative_dir_index_list = []
            self.label_list = []
            for index, img_dir in enumerate(self.relative_dir_list):
                dir_list = os.listdir(f"{self.root_dir}/{img_dir}")
                self.img_name_list += dir_list
                self.relative_dir_index_list += [index] * len(dir_list)
                self.label_list += [index] * len(dir_list)

            self.transform1 = transforms.Compose([  # 串联多个图片变换的操作
                transforms.Resize(256),  # 缩放
                transforms.CenterCrop(224),  # 中心裁剪
                transforms.ToTensor()]  # 转换成Tensor
            )
            resnet50_feature_extractor = tv_models.resnet50(pretrained=True)  # 导入ResNet50的预训练模型
            # resnet50_feature_extractor.fc = nn.Linear(2048, 2048)  # 重新定义最后一层
            # torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  # 将二维tensor初始化为单位矩阵
            # for param in resnet50_feature_extractor.parameters():
            #     param.requires_grad = False
            # self.feature_extractor = resnet50_feature_extractor
            resnet50_feature_extractor.eval()
            # resnet50_feature_extractor = resnet50_feature_extractor.cuda()
            self.feature_extractor = resnet50_feature_extractor

        def __getitem__(self, index):
            img = PIL.Image.open(f"{self.root_dir}/{self.relative_dir_list[self.relative_dir_index_list[index]]}/{self.img_name_list[index]}")  # 打开图片
            img1 = self.transform1(img)  # 对图片进行transform1的各种操作
            x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)

            with torch.no_grad():
                image_features = self.feature_extractor(x)
            target_label_tensor = torch.tensor(self.label_list[index])
            return image_features, target_label_tensor

        def __len__(self):
            return len(self.img_name_list)

        def get_name_list(self):
            return self.img_name_list

    def __init__(self, root_dir, pkl_name="adnc_features", out_dims=64, task_lim=1, aliquot=3*5, cuda=True):
        self.name = pkl_name
        # 读取pkl文件，整理tensor，stacked[-1]是标签值
        with open(f'{root_dir}/{pkl_name}.pkl', 'rb') as f:
            stacked = pickle.load(f)
        stacked[0] = stacked[0][:, 0, :]  # 提前选好设备
        stacked[1] = stacked[1].unsqueeze(1)
        stacked = [me.cuda() if cuda else me.cpu() for me in stacked]

        # 乘以task_lim种分类
        output_tran = stacked[-1].transpose(0, 1)
        labels = [set(output_line.tolist()) for output_line in output_tran]
        self.all_labels = [list(lab) for _ in range(task_lim) for lab in labels]

        # 将相邻的task_lim个样本缝在一起 每个样本都有task_lim个分类
        sample_count = stacked[0].shape[0] // task_lim  # 丢弃剩余除不尽的部分
        for i, me in enumerate(stacked):
            me = me[:sample_count * task_lim]
            stacked[i] = me.reshape(me.shape[0]//task_lim, me.shape[1]*task_lim)

        # 降维
        if out_dims > 0:
            print("features' shape before lstm:", [me.shape for me in stacked[:-1]])
            for i, me in enumerate(stacked[:-1]):
                me_lstm = nn.LSTM(input_size=me.shape[1],
                                  hidden_size=int(me.shape[1] * out_dims / 2) if isinstance(out_dims, float) else out_dims // 2,
                                  num_layers=2, bidirectional=True, dropout=0.5)
                if cuda:
                    me_lstm = me_lstm.cuda()
                me_lstm.requires_grad_(False)
                stacked[i] = me_lstm(me)[0]
            print("features' shape after lstm:", [me.shape for me in stacked[:-1]])

        # 样本返回格式
        self.data = []
        for i in range(sample_count):  # i：样本数
            self.data.append(tuple(stacked[j][i] for j in range(len(stacked))))  # j：特征数
        self.data = self.data[:len(self.data) // aliquot * aliquot]  # 去除部分样本 使其能被batch_size整除且按比例划分数据集

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # pkl_data_set = ADNCDataset.Pkl("../data/ADCN")
    # pickle_data(pkl_data_set, pkl_data_set.root_dir, pkl_data_set.img_dir_list,
    #             [pkl_data_set.excel_file.values[i][0] for i in range(len(pkl_data_set))])

    pkl_data_set = BloodCellDataset.Pkl("../data/blood-cells/dataset2-master/images/TEST")
    pickle_data(pkl_data_set, pkl_data_set.root_dir,
                pkl_data_set.relative_dir_list, pkl_data_set.img_name_list, "bloodcells_feat")
