import pandas as pd
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv('SAT_Questions_multiple_choice.csv')

# 存储转换后的数据
data = []
train_data = []
validation_data = []

for index, row in df.iterrows():
    question = row['Question']
    possible_answers = row['Possible Answers'].split(' ')  # 可能的答案
    correct_answer = row['Answer']  # 正确答案索引

    # 对于每个问题，先处理正确答案
    correct_index = ord(correct_answer) - ord('A')  # 获取正确答案的索引
    correct_statement = f"{question} {possible_answers[correct_index]}"
    data.append({
        'question': correct_statement,
        'answer': True,
        'passage': question
    })

    # 然后处理所有错误答案
    wrong_choices = [i for i in range(len(possible_answers)) if i != correct_index]
    for wrong_index in wrong_choices:
        wrong_statement = f"{question} {possible_answers[wrong_index]}"
        data.append({
            'question': wrong_statement,
            'answer': False,
            'passage': question
        })

df_data = pd.DataFrame(data)
train_data, validation_data = train_test_split(df_data, test_size=0.1, random_state=42)

# 将train_data和validation_data转换为Dataset格式
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)

# 创建DatasetDict对象
final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

# 输出结果
final_dataset.save_to_disk('SAT_dataset')

print("转换完成，数据已保存到 final_dataset 文件夹")
