import sys,base64,json,subprocess,os
p=sys.argv[1]; name=os.path.basename(p)
path='train-bridge-loop/'+name
body={"message":"Add train bridge loop video part "+name+"\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011eudubV1NH8SujXCENxsKQ","branch":"main","content":base64.b64encode(open(p,'rb').read()).decode()}
j=p+'.json'; json.dump(body,open(j,'w'))
r=subprocess.run(['gh','api','-X','PUT','repos/lewisjakewhite26/C4comp/contents/'+path,'--input',j],capture_output=True,text=True)
os.remove(j)
print(r.returncode,r.stdout[:200],r.stderr[:300])
sys.exit(r.returncode)
