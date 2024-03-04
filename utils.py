
from zhipuai import ZhipuAI
import json

def chatGLM(prompt):
    api_key=""
    client = ZhipuAI(api_key=api_key) # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
  
# 将大模型总结保存到json 文件中
def llm_res_json(CEs,header_row,consum,input_data,file_name):
    if len(CEs)>0:
        prompt_blank="I will provide you with input and output data for the model. The input data for the model includes the following features: ‘{}‘. The model predicts the output data and label ‘{}’ based on the input data. Please generate a text explanation on how to change the predicted label from input data to output data. The output text should be in the format: ‘To change the predicted label from label_1 to label_0, we need to decrease/increase the value of [feature_name] to [new_value].’ Input data:{} Output data:{}"
        list=[]
        for index in range(0,len(CEs)):
            output_data = CEs.iloc[index].values
            prompt=prompt_blank.format(header_row,consum,input_data,output_data) # 构造完整的prompt
            res=chatGLM(prompt)
            list.append({
                "input":str(input_data),
                "output":str(output_data),
                "result":res
            })

        with open("./Hour/effi/"+file_name+".json", 'w') as file:
            json.dump(list, file)  # 使用json.dump()函数将数据写入文件中
        