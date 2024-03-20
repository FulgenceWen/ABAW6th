# import torch

# from timm import create_model


# backbone = create_model(
#                 'timm/efficientnetv2_rw_t.ra2_in1k',
#                 pretrained=False,
#                 num_classes=0,  # remove classifier nn.Linear
#                 # features_only=True,
#                 # global_pool='',
#             )

# image = torch.randn(1, 3, 112, 112)
# output = backbone(image)
# print(output.shape)

# # for name, _ in backbone.named_parameters():
#     # print(name)
# # 
# # ckpt_path = "/data/zhangfengyu/ABAWcodes/mycode/model/mae_face_pretrain_vit_base.pth"
# # state_dict = torch.load(ckpt_path)["model"]
# # 
# # 
# # 
# # print(state_dict[])

list1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
list2 = ['a', 'c', 'e', 'g']

aligned_list = []  # 创建一个空的对齐列表

index_list2 = 0  # 列表2的索引

# 遍历列表1中的元素
for item in list1:
    # 如果列表2的索引小于列表2的长度且当前列表2的元素等于列表1的元素
    # 则将该元素添加到对齐列表中，并将列表2的索引加1
    if index_list2 < len(list2) and list2[index_list2] == item:
        aligned_list.append(item)
        index_list2 += 1
    else:
        # 如果列表2的索引大于等于列表2的长度或当前列表2的元素不等于列表1的元素
        # 则将列表1的元素添加到对齐列表中
        aligned_list.append(item)

print(aligned_list)
