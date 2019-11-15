# bert-django:blush:
 对bert的一个移植测试，停供了bert 对中文的向量接口和命名实体识别的接口，希望对的学习bert 的朋友有点帮助。

## env version
  tenorflow-gpu == 1.13.1
  djangorestframework == 3.9.2
  
## data model 
    bert chinese model[chinese_L-12_H-768_A-12](https://github.com/google-research/bert)
    bert NER model [bert-ner](https://pan.baidu.com/s/10VYvMN24O7rnyaM-__P0wA)

## usage
  ## start :
  python  manage.py runserver 192.168.1.105:8888
  
  ## request protocol
  id : 192.168.1.105:8888
  vector post request json:
    {
      "method":"bertvec",
      "content":"习近平：止暴制乱 恢复秩序是香港当前最紧迫的任务"
    }
  result:
    {
        "result": reshape（1,768）, // 返回一个 shape为（1,768）的向量
        "cost_time": "0.24736428260803223"
    }
  
  ner post request protocol
    {
      "method":"bertNER",
      "content":"习近平：止暴制乱 恢复秩序是香港当前最紧迫的任务"
    }
  result: //返回实体名称
    {
        "result": {
            "习近平": "B-PER",  
            "香港": "B-LOC"
        },
        "cost_time": "0.6205282211303711"
    }
    
    
