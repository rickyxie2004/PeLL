import pandas as pd
from datasets import Dataset, DatasetDict

# 读取Parquet文件
df = pd.read_parquet('arxiv.parquet')

# 重构数据格式
df_data = pd.DataFrame({
    'passage': df['question'],
    'question': df['answer'],
    'answer': True  # 统一标记为True
})

# 使用train_test_split分割数据集，90%作为训练集，10%作为验证集
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(df_data, test_size=0.1, random_state=42)

# 转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)
train_data = train_data.reset_index(drop=True)  # 删除索引列
validation_data = validation_data.reset_index(drop=True)  # 删除索引列

# 创建 DatasetDict
final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

# 保存数据集
final_dataset.save_to_disk('arxiv_dataset')

print("转换完成，数据已保存到 final_dataset 文件夹")
