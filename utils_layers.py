import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
from hornet import MatchingAttention
from prompt import Prompt


class GeomGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)  # 修改 移除.cuda()

        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)

        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.merge = merge

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        tmp = g.number_of_edges()
        for i in range(g.number_of_edges()):
            if i % 1000 == 0:
                print(f"\r{i + 1}/{tmp}", end="", flush=True)
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])
        print(f"\r{tmp}/{tmp}")
        # subgraph_edge_list 为节点关系的list：9 list[0]-list[9]
        return subgraph_edge_list

    def forward(self, feature):
        self.g.ndata['h'] = self.in_feats_dropout(feature)  # Output is of the same shape as input
        for i in range(self.num_divisions):  # num_divisions为关系个数，论文设置为9
            subg = self.g.edge_subgraph(self.subgraph_edge_list_of_list[i])
            subg.ndata[f'Wh_{i}'] = self.linear_for_each_division[i](subg.ndata['h']) * subg.ndata['norm']
            subg.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),
                            reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subg.ndata.pop(f'Wh_{i}')
            self.g.nodes[subg.ndata['_ID']].data[f'h_{i}'] = subg.ndata[f'h_{i}']
        self.g.ndata.pop('h')

        if self.merge == 'cat':
            # (batchsize*509, out_feats*num_divisions)
            return self.activation(torch.cat([self.g.ndata.pop(f'h_{i}') for i in range(self.num_divisions)],
                                             dim=-1) * self.g.ndata['norm'])
        else:
            # (batchsize*509, out_feats)
            return self.activation(
                torch.mean(torch.stack([self.g.ndata.pop(f'h_{i}') for i in range(self.num_divisions)],
                                       dim=-1), dim=-1) * self.g.ndata['norm'])


class GeomGCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge):
        super(GeomGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions, activation, dropout_prob, ggcn_merge))
        self.channel_merge = channel_merge

    def forward(self, feature):
        # list里有num_heads个（batchsize*509, out_feats*num_divisions）
        if self.channel_merge == 'cat':
            # (batchsize*509, out_feats*num_divisions*num_heads)
            return torch.cat([head(feature) for head in self.attention_heads], dim=1)
        else:
            # (batchsize*509, out_feats)
            return torch.mean(torch.stack([head(feature) for head in self.attention_heads]), dim=0)


class GeomGCNNet(nn.Module):
    def __init__(self, g, num_input_features, num_output_classes, num_hidden, num_divisions, num_heads_layer_one,
                 num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, layer_two_ggcn_merge,
                 layer_two_channel_merge, sorts, batch_size, m_prompt, m_matchatt,
                 head_type='token', use_prompt_mask=False):
        super(GeomGCNNet, self).__init__()
        self.head_type = head_type  # 分类头的输入类型
        self.use_prompt_mask = use_prompt_mask
        self.prompt = m_prompt()
        # self.prompt = nn.Linear(g.num_nodes(), g.num_nodes())  # base + Liner + 门控卷积
        print("======== 2.1 ========")
        self.geomgcn1 = GeomGCN(g, 1, num_hidden, num_divisions, F.relu, num_heads_layer_one,
                                dropout_rate,
                                layer_one_ggcn_merge, layer_one_channel_merge)
        print("======== 2.2 ========")
        self.geomgcn2 = GeomGCN(g, num_hidden * num_divisions * num_heads_layer_one,
                                num_output_classes, num_divisions, lambda x: x,
                                num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge)
        self.g = g
        self.batch_size = batch_size
        # 10多分类
        self.output_layers = nn.ModuleList()
        self.sorts = sorts
        for sort in sorts:
            self.output_layers.append(nn.Linear(g.num_nodes() * num_output_classes,
                                                batch_size * len(sort)))
        self.matchatt = m_matchatt()
        # self.matchatt = nn.Linear(g.num_nodes(), g.num_nodes())  # base + prompt + Liner

    def forward_features(self, x, task_id=-1, train=False):
        if self.prompt is None:
            return x

        if self.use_prompt_mask and train:
            start = task_id * self.prompt.top_k
            end = (task_id + 1) * self.prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(x.device)
            # 生成一个start到end(不包括end）的等差一维张量 例如：torch.arange(0,5) = tensor([0, 1, 2, 3, 4])
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)  # 复制batchsize份
            if end > self.prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        x = self.prompt(x, prompt_mask=prompt_mask)
        return x

    def forward(self, features, task_id=-1, train=False):  # (509*batch_size, 1)
        # 在该文件中，对包括这里的许多地方进行了压缩表示，除此之外代码逻辑没有任何变动，本意是想提高速度，但是好像没多大起色
        # x = self.geomgcn2(  # base
        #     self.geomgcn1(features.view(-1).unsqueeze(1))).view(-1)
        # x = self.geomgcn2(  # base + 门控卷积
        #     self.geomgcn1(
        #         self.matchatt(features.view(-1).unsqueeze(1)))).view(-1)
        # x = self.geomgcn2( # base + prompt
        #     self.geomgcn1(
        #         self.forward_features(features, task_id, train).view(-1).unsqueeze(1))).view(-1)
        # x = self.geomgcn2(  # base + 门控卷积 + prompt
        #     self.geomgcn1(
        #         self.forward_features(
        #             self.matchatt(features.view(-1).unsqueeze(1)).view(features.shape[0], features.shape[1]), task_id, train).view(-1).unsqueeze(1))).view(-1)
        x = self.geomgcn2(  # base + prompt + 门控卷积
            self.geomgcn1(
                self.matchatt(
                    self.forward_features(features, task_id, train).view(-1).unsqueeze(1)))).view(-1)
        # x = self.geomgcn2(  # base + prompt + Liner
        #     self.geomgcn1(
        #         self.matchatt(
        #             self.forward_features(features, task_id, train).view(-1)).unsqueeze(1))).view(-1)
        # x = self.geomgcn2(  # base + Liner + 门控卷积
        #     self.geomgcn1(
        #         self.matchatt(
        #             self.prompt(features.view(-1)).unsqueeze(1)))).view(-1)
        return [output_layer(x).view(self.batch_size, len(sort)) for output_layer, sort in zip(self.output_layers, self.sorts)]
