#!/usr/bin/env python3
"""Beam search decoder for sequence generation."""
import sys,math

def beam_search(score_fn,vocab,start,end,max_len=20,beam_width=3):
    beams=[(0.0,[start])]
    for _ in range(max_len):
        candidates=[]
        for score,seq in beams:
            if seq[-1]==end:candidates.append((score,seq));continue
            scores=score_fn(seq,vocab)
            for tok,s in scores:
                candidates.append((score+s,seq+[tok]))
        candidates.sort(key=lambda x:-x[0])
        beams=candidates[:beam_width]
        if all(b[1][-1]==end for b in beams):break
    return beams

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        vocab=['a','b','c','<end>']
        # Score function that prefers 'a' then 'b' then '<end>'
        def score_fn(seq,v):
            step=len(seq)-1
            scores=[]
            for tok in v:
                if step==0:s=0.9 if tok=='a' else 0.01
                elif step==1:s=0.8 if tok=='b' else 0.05
                else:s=0.9 if tok=='<end>' else 0.01
                scores.append((tok,math.log(s+1e-10)))
            return scores
        results=beam_search(score_fn,vocab,'<start>','<end>',beam_width=3)
        assert len(results)>0
        best=results[0][1]
        assert best==['<start>','a','b','<end>'],f"Got {best}"
        print("All tests passed!")
    else:
        vocab=['the','cat','sat','<end>']
        def score_fn(seq,v):
            order={'the':0.9,'cat':0.7,'sat':0.5,'<end>':0.3}
            return[(t,math.log(order.get(t,0.01)))for t in v]
        results=beam_search(score_fn,vocab,'<start>','<end>')
        for s,seq in results:print(f"  {s:.2f}: {' '.join(seq)}")
if __name__=="__main__":main()
