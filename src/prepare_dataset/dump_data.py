import pandas as pd
import json
import random
from datasets import Dataset, DatasetDict

class_name = "high_school"
# 读取Parquet文件
df = pd.read_parquet(f'{class_name}.parquet')

# 存储转换后的数据
data_samples = []

for index, row in df.iterrows():
    question = row['question']
    choices = row['choices']
    correct_index = int(row['answer'])  # 直接转换为整数  # 获取正确答案索引

    statement = f"{question} {choices[correct_index]}"
    label = True
    data_samples.append({
        "question": statement,
        "answer": label,
        "passage": question  # passage与question一致
    })

    '''
    wrong_choices = [i for i in range(len(choices)) if i != correct_index]
    # print(wrong_choices)
    for wrong_index in wrong_choices:
        statement = f"{question} {choices[wrong_index]}"
        label = "False"
        # 组织数据格式
        data_samples.append({
            "question": statement,
            "answer": label,
            "passage": question  # passage与question一致
        })
    '''
    

# 转换为 Hugging Face Dataset 格式
dataset = Dataset.from_list(data_samples)

dataset_dict = DatasetDict({
    "train": dataset.select(range(int(0.9 * len(dataset)))),  # 90% 训练集
    "validation": dataset.select(range(int(0.9 * len(dataset)), len(dataset)))  # 10% 验证集
})

# 保存数据集
dataset_dict.save_to_disk(f"{class_name}_dataset")
print(f"转换完成，数据已保存到 {class_name} dataset")