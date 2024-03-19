from transformers import BartTokenizer, BartForConditionalGeneration
import os

proxies = {
    'http': 'http://edcguest:edcguest@172.31.100.25:3128',
    'https': 'http://edcguest:edcguest@172.31.100.25:3128'
}

os.environ['http_proxy'] = proxies['http']
os.environ['https_proxy'] = proxies['https']

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name, proxies=proxies)
model = BartForConditionalGeneration.from_pretrained(model_name, proxies=proxies)