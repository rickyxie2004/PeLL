import os
from datasets import load_from_disk, DatasetDict, concatenate_datasets

# 获取当前目录下所有以 "dataset" 结尾的文件夹
dataset_dirs = [d for d in os.listdir('.') if d.endswith('dataset') and os.path.isdir(d)]

# 存储加载的数据集
datasets_list = []

# 逐个加载数据集并转换 answer 字段类型
for dataset_dir in dataset_dirs:
    # print(f"Dealing {dataset_dir}")
    dataset = load_from_disk(dataset_dir)

    def cast_answer_to_bool(example):
        example["answer"] = example["answer"] == True or example["answer"] == "True"
        return example

    # 确保 answer 是 bool 类型
    dataset = dataset.map(cast_answer_to_bool)

    datasets_list.append(dataset)

# 确保数据集格式一致
if not datasets_list:
    print("未找到任何 dataset 结尾的文件夹")
else:
    # 按照 train 和 validation 进行合并
    merged_train = concatenate_datasets([ds['train'] for ds in datasets_list])
    merged_validation = concatenate_datasets([ds['validation'] for ds in datasets_list])

    # 创建合并后的 DatasetDict
    merged_dataset = DatasetDict({
        'train': merged_train,
        'validation': merged_validation
    })

    # 保存合并后的数据集
    merged_dataset.save_to_disk('merged_dataset')

    print(f"合并完成，共合并 {len(datasets_list)} 个数据集，数据已保存到 merged_dataset 目录")
