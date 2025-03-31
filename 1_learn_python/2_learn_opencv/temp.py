from transformers import BertModel, BertTokenizer


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentences = 'Give me a book.',
input_ids = tokenizer.encode(sentences, padding=True, return_tensors='pt')
outputs = model(input_ids, output_attentions=True)
attentions = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

from bertviz import head_view

head_view(attentions, tokens)
