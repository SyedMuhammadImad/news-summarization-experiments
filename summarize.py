"""BART news summarization with input truncation, generation and ROUGE scoring."""
import argparse
import json
from pathlib import Path
import re

DEFAULT_MODEL='facebook/bart-large-cnn'

def clean_text(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Article must be non-empty text.')
    return re.sub(r'\s+',' ',text).strip()

def load_model(name=DEFAULT_MODEL):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer=AutoTokenizer.from_pretrained(name,token=False,trust_remote_code=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(name,token=False,trust_remote_code=False)
    model.eval()
    return tokenizer,model

def generate_summary(tokenizer,model,article,max_new_tokens=128):
    import torch
    if not 1 <= max_new_tokens <= 512:raise ValueError('Output limit must be between 1 and 512 tokens.')
    article=clean_text(article)
    max_input=min(int(getattr(tokenizer,'model_max_length',1024)),1024)
    inputs=tokenizer(article,max_length=max_input,truncation=True,return_tensors='pt')
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=max_new_tokens,num_beams=2,
                                 do_sample=False,early_stopping=True)
    summary=tokenizer.decode(generated[0],skip_special_tokens=True).strip()
    if not summary:raise RuntimeError('Model returned an empty summary.')
    return summary

def evaluate(tokenizer,model,rows,max_new_tokens=128):
    from rouge_score import rouge_scorer
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    results=[]
    for row in rows:
        summary=generate_summary(tokenizer,model,row['article'],max_new_tokens)
        scores=scorer.score(row['highlights'],summary)
        results.append({'summary':summary,**{k:v.fmeasure for k,v in scores.items()}})
    if not results:raise ValueError('Evaluation requires at least one article.')
    return {'count':len(results),'rouge':{k:sum(r[k] for r in results)/len(results) for k in ('rouge1','rouge2','rougeL')},'predictions':results}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True,help='CNN/DailyMail test or validation CSV with article,highlights')
    parser.add_argument('--model',default=DEFAULT_MODEL)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--output',type=Path,default=Path('summarization_metrics.json'))
    args=parser.parse_args()
    if args.limit<=0:parser.error('--limit must be positive')
    import pandas as pd
    frame=pd.read_csv(args.data,nrows=args.limit)
    if not {'article','highlights'}<=set(frame.columns):parser.error('CSV must contain article and highlights')
    tokenizer,model=load_model(args.model)
    result=evaluate(tokenizer,model,frame.to_dict('records'),args.max_new_tokens)
    args.output.write_text(json.dumps({'model':args.model,'dataset':'CNN/DailyMail supplied evaluation CSV','rows':'first N; no training performed',**result},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'}))

if __name__=='__main__':main()
