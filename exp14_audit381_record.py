#!/usr/bin/env python3
"""Verify and record an existing editorial audit; does not rewrite or re-audit the article."""
from pathlib import Path
import hashlib,json,re,datetime
R=Path(__file__).resolve().parent
ARTICLE=R.parent/'ai-theorys-study/wechat/381.md'
REVIEW=ARTICLE.parent/'assets/381/审稿意见.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 initial=json.loads((R/'results/module16_initial_article.json').read_text());assert sha(ARTICLE)==initial['sha256_before'],'Article changed: rerun the editorial audit rather than silently reusing it'
 if not REVIEW.is_file():raise RuntimeError('Final review is not written yet; complete module16 editorial review first')
 text=REVIEW.read_text();assert len(text)>1000,'Audit appears incomplete'
 if initial['sha256_before'] not in text:raise RuntimeError('Review must record the unchanged article SHA256')
 urls=sorted({v.rstrip('*.,，。;；') for v in re.findall(r'https?://[^\s<>）)]+',ARTICLE.read_text())})
 result={'module':16,'status':'AUDIT_COMPLETED_WITH_FINDINGS','article_path':str(ARTICLE),'review_path':str(REVIEW),'article_sha256_before':initial['sha256_before'],'article_sha256_after':sha(ARTICLE),'article_modified':False,'review_sha256':sha(REVIEW),'review_characters':len(text),'article_unique_urls':urls,'scope':'Existing full editorial audit, including numerical/factual evidence and URL limitations; this CLI verifies artifact identity only, not a new LLM or web review','body_remains_with_findings':True,'timestamp':datetime.datetime.now().astimezone().isoformat()}
 p2=text.split('## P2：',1)[1].split('## 数值核验账本',1)[0]
 result['findings_by_severity']={'P0':len(re.findall(r'^### P0-\d+',text,re.M)),'P1':len(re.findall(r'^### P1-\d+',text,re.M)),'P2':len(re.findall(r'^\d+\. ',p2,re.M))}
 assets=sorted((ARTICLE.parent/'assets/381').glob('*.svg'))+[ARTICLE.parent/'cover381.svg']
 result['protected_assets']=[{'path':str(a),'sha256':sha(a),'hash_recorded_in_review':sha(a) in text} for a in assets]
 assert len(assets)==6 and all(a['hash_recorded_in_review'] for a in result['protected_assets'])
 result['remote_URL_limitations']='4 GitHub pages content-verified; 3 WeChat URLs return CAPTCHA despite HTTP200, remote article content not confirmed'
 (R/'results/module16_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
