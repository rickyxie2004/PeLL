import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# 指定包含JSON文件的目录
folder_path = 'camel'

# 存储转换后的数据
data = []

# 遍历camel文件夹中的所有JSON文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            # 提取message_1和message_2
            message_1 = content.get('message_1', '').strip()
            message_2 = content.get('message_2', '').strip()

            # 构建样本，passage为message_1，question为message_2，answer为True
            if message_1 and message_2:
                data.append({
                    'passage': message_1,
                    'question': message_2,
                    'answer': True
                })

# 将数据转换为DataFrame
df_data = pd.DataFrame(data)

# 使用train_test_split分割数据集，90%作为训练集，10%作为验证集
train_data, validation_data = train_test_split(df_data, test_size=0.1, random_state=42)

# 删除DataFrame中的索引列（如果存在）
train_data = train_data.reset_index(drop=True)  # 删除索引列
validation_data = validation_data.reset_index(drop=True)  # 删除索引列

# 将train_data和validation_data转换为Dataset格式
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)

# 创建DatasetDict对象
final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

# 输出结果
final_dataset.save_to_disk('camel_dataset')

print("转换完成，数据已保存到 final_dataset 文件夹")
