from datasets import load_from_disk

# 读取保存的数据集
dataset_dict = load_from_disk("merged_dataset")

# 查看数据集结构
print(dataset_dict)

# 访问训练集和验证集
train_dataset = dataset_dict["train"]
validation_dataset = dataset_dict["validation"]

# 查看训练集前几条数据
print(train_dataset[0])